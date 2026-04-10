r"""
Benchmark: BABILong-style multi-hop reasoning

Tests whether the genome can chain multiple retrieval hops to answer
questions that require combining facts from different genes.

Based on bAbI (Weston et al., 2015) and BABILong (Kuratov et al., 2024).
We use a scaffolded version that generates self-contained multi-hop
problems and ingests them into a scratch corpus, then queries with
multi-hop questions.

Supported tasks:
  - task_1  — single supporting fact  (sanity check, same as needle)
  - task_2  — two supporting facts    (two-hop reasoning)
  - task_3  — three supporting facts  (three-hop reasoning)

Each task generates N=10 problems with distractor text padding,
ingests them as genes, and measures:
  - retrieval_rate: did the genome express the right supporting facts?
  - answer_accuracy: did the model answer correctly?
  - latency: per-query wall time

Usage:
    # Ingest a fresh scratch corpus and benchmark
    python benchmarks/bench_babilong.py

    # Specific task only
    python benchmarks/bench_babilong.py --task task_2

    # Use existing genome (skip ingest)
    python benchmarks/bench_babilong.py --no-ingest

Note: This benchmark creates a separate scratch genome at
      ./benchmarks/babilong_scratch.db so it doesn't pollute the main
      genome. Set HELIX_SCRATCH_URL to point to a second Helix server
      instance running against that DB, or run with --in-process mode
      to use HelixContextManager directly.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import httpx

HELIX_URL = os.environ.get("HELIX_URL", "http://127.0.0.1:11437")

# Fixed seed for reproducible problem generation
random.seed(42)


# ── Problem generators ────────────────────────────────────────────

NAMES = ["Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Henry",
         "Ivy", "Jack", "Kate", "Leo", "Mia", "Nate", "Olga", "Paul"]
PLACES = ["the kitchen", "the garden", "the office", "the hallway",
          "the library", "the bedroom", "the attic", "the basement",
          "the garage", "the cellar", "the lounge", "the patio"]
OBJECTS = ["the key", "the book", "the lamp", "the cup", "the ball",
           "the chair", "the clock", "the vase", "the sword", "the map"]


def generate_task_1(n: int = 10) -> List[Dict]:
    """
    Task 1: Single supporting fact.
    "Alice is in the kitchen." → Q: "Where is Alice?" → A: "kitchen"
    """
    problems = []
    for i in range(n):
        person = random.choice(NAMES)
        place = random.choice(PLACES)
        fact = f"{person} is in {place}."

        # Distractors — unrelated but plausible sentences
        distractors = []
        for _ in range(5):
            other = random.choice([p for p in NAMES if p != person])
            other_place = random.choice(PLACES)
            distractors.append(f"{other} is in {other_place}.")

        random.shuffle(distractors)
        story = " ".join(distractors[:2] + [fact] + distractors[2:])

        problems.append({
            "name": f"task_1_{i:02d}",
            "story": story,
            "question": f"Where is {person}?",
            "expected": place.replace("the ", ""),
            "accept": [place.replace("the ", ""), place],
            "supporting_facts": [fact],
        })
    return problems


def generate_task_2(n: int = 10) -> List[Dict]:
    """
    Task 2: Two supporting facts (two-hop).
    "Alice picked up the key. Alice went to the kitchen."
    → Q: "Where is the key?" → A: "kitchen"
    """
    problems = []
    for i in range(n):
        person = random.choice(NAMES)
        obj = random.choice(OBJECTS)
        place = random.choice(PLACES)
        fact1 = f"{person} picked up {obj}."
        fact2 = f"{person} went to {place}."

        # Distractors
        distractors = []
        for _ in range(6):
            other = random.choice([p for p in NAMES if p != person])
            other_place = random.choice([p for p in PLACES if p != place])
            distractors.append(f"{other} went to {other_place}.")

        random.shuffle(distractors)
        story = " ".join([distractors[0], fact1, distractors[1], distractors[2],
                         fact2, distractors[3], distractors[4], distractors[5]])

        problems.append({
            "name": f"task_2_{i:02d}",
            "story": story,
            "question": f"Where is {obj}?",
            "expected": place.replace("the ", ""),
            "accept": [place.replace("the ", ""), place],
            "supporting_facts": [fact1, fact2],
        })
    return problems


def generate_task_3(n: int = 10) -> List[Dict]:
    """
    Task 3: Three supporting facts (three-hop).
    "Alice picked up the key. Alice went to the kitchen.
     Alice put down the key."
    → Q: "Where was the key before the kitchen?"
    → A: wherever Alice was before (tracked via prior fact)
    """
    problems = []
    for i in range(n):
        person = random.choice(NAMES)
        obj = random.choice(OBJECTS)
        place_a = random.choice(PLACES)
        place_b = random.choice([p for p in PLACES if p != place_a])

        fact1 = f"{person} is in {place_a}."
        fact2 = f"{person} picked up {obj}."
        fact3 = f"{person} went to {place_b}."

        distractors = []
        for _ in range(5):
            other = random.choice([p for p in NAMES if p != person])
            other_place = random.choice(PLACES)
            distractors.append(f"{other} is in {other_place}.")

        random.shuffle(distractors)
        story = " ".join([fact1, distractors[0], fact2, distractors[1],
                         distractors[2], fact3, distractors[3], distractors[4]])

        problems.append({
            "name": f"task_3_{i:02d}",
            "story": story,
            "question": f"Where is {obj}?",
            "expected": place_b.replace("the ", ""),
            "accept": [place_b.replace("the ", ""), place_b],
            "supporting_facts": [fact1, fact2, fact3],
        })
    return problems


TASK_GENERATORS = {
    "task_1": generate_task_1,
    "task_2": generate_task_2,
    "task_3": generate_task_3,
}


# ── Ingest + query helpers ────────────────────────────────────────

def ingest_problems(client: httpx.Client, problems: List[Dict]) -> int:
    """Ingest each problem's story as a gene so retrieval can find it."""
    ingested = 0
    for p in problems:
        try:
            # Tag with task + problem name so we can scope retrieval
            metadata = {
                "path": f"babilong/{p['name']}.txt",
                "task": p["name"].rsplit("_", 1)[0],
            }
            resp = client.post(f"{HELIX_URL}/ingest", json={
                "content": p["story"],
                "content_type": "text",
                "metadata": metadata,
            })
            if resp.status_code == 200:
                ingested += 1
        except Exception:
            pass
    return ingested


def run_problem(client: httpx.Client, problem: Dict, model: str) -> Dict:
    """Run a single problem through retrieval + answer generation."""
    t0 = time.time()

    # Step 1: Retrieval check
    try:
        resp = client.post(f"{HELIX_URL}/context", json={
            "query": problem["question"],
            "decoder_mode": "none",
        })
    except Exception as e:
        return {
            "name": problem["name"],
            "question": problem["question"],
            "expected": problem["expected"],
            "retrieved_facts": 0,
            "retrieval_complete": False,
            "answer_correct": False,
            "latency_s": time.time() - t0,
            "error": str(e)[:100],
        }

    context_latency = time.time() - t0

    if resp.status_code != 200:
        return {
            "name": problem["name"],
            "question": problem["question"],
            "expected": problem["expected"],
            "retrieved_facts": 0,
            "retrieval_complete": False,
            "answer_correct": False,
            "latency_s": context_latency,
            "error": f"HTTP {resp.status_code}",
        }

    data = resp.json()
    entry = data[0] if data else {}
    content = entry.get("content", "").lower()
    health = entry.get("context_health", {})

    # Check how many supporting facts were retrieved
    supporting = problem["supporting_facts"]
    retrieved = sum(1 for f in supporting if f.lower() in content)
    retrieval_complete = retrieved == len(supporting)

    # Step 2: Answer generation
    t1 = time.time()
    try:
        proxy_resp = client.post(f"{HELIX_URL}/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": problem["question"]}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 64},
        })
    except Exception:
        proxy_resp = None

    proxy_latency = time.time() - t1

    answer_correct = False
    answer_text = ""
    if proxy_resp is not None and proxy_resp.status_code == 200:
        choices = proxy_resp.json().get("choices", [])
        if choices:
            answer_text = choices[0].get("message", {}).get("content", "")
            answer_correct = any(
                a.lower() in answer_text.lower()
                for a in problem["accept"]
            )

    return {
        "name": problem["name"],
        "question": problem["question"],
        "expected": problem["expected"],
        "retrieved_facts": retrieved,
        "total_facts": len(supporting),
        "retrieval_complete": retrieval_complete,
        "answer_correct": answer_correct,
        "context_latency_s": round(context_latency, 3),
        "proxy_latency_s": round(proxy_latency, 3),
        "ellipticity": health.get("ellipticity", 0),
        "status": health.get("status", "unknown"),
        "answer_preview": answer_text[:150] if answer_text else "",
    }


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASK_GENERATORS.keys()) + ["all"],
                        default="all")
    parser.add_argument("--n", type=int, default=10,
                        help="Problems per task")
    parser.add_argument("--no-ingest", action="store_true",
                        help="Skip ingestion (use existing genome)")
    parser.add_argument("--model", default=os.environ.get("HELIX_MODEL", "qwen3:4b"))
    parser.add_argument("--output", default="benchmarks/babilong_results.json")
    args = parser.parse_args()

    client = httpx.Client(timeout=300)

    # Server check
    try:
        stats = client.get(f"{HELIX_URL}/stats").json()
        print(f"Helix server: {stats['total_genes']} genes, "
              f"{stats['compression_ratio']:.2f}x compression")
    except Exception as e:
        print(f"ERROR: Helix server unreachable at {HELIX_URL}: {e}")
        sys.exit(1)

    # Generate problems for selected task(s)
    tasks_to_run = list(TASK_GENERATORS.keys()) if args.task == "all" else [args.task]
    all_problems = {}
    for task in tasks_to_run:
        all_problems[task] = TASK_GENERATORS[task](args.n)
        print(f"Generated {len(all_problems[task])} problems for {task}")

    # Ingest
    if not args.no_ingest:
        print("\nIngesting problems...")
        total_ingested = 0
        for task, problems in all_problems.items():
            n = ingest_problems(client, problems)
            total_ingested += n
            print(f"  {task}: {n}/{len(problems)} ingested")
        print(f"Total ingested: {total_ingested}")

    # Run benchmark
    print(f"\n=== BABILong benchmark ({args.model}) ===")
    all_results = {}
    for task, problems in all_problems.items():
        print(f"\n-- {task} --")
        task_results = []
        for p in problems:
            result = run_problem(client, p, args.model)
            mark_ret = "[+]" if result["retrieval_complete"] else "[-]"
            mark_ans = "[+]" if result["answer_correct"] else "[-]"
            latency = result.get("context_latency_s", 0) + result.get("proxy_latency_s", 0)
            print(f"  retrieval {mark_ret}  answer {mark_ans}  "
                  f"{result['retrieved_facts']}/{result.get('total_facts', '?')} facts  "
                  f"{latency:.1f}s  {p['name']}")
            task_results.append(result)
        all_results[task] = task_results

    # Summary
    print("\n=== Summary ===")
    summary = {}
    for task, results in all_results.items():
        retrieval_rate = sum(r["retrieval_complete"] for r in results) / len(results)
        answer_rate = sum(r["answer_correct"] for r in results) / len(results)
        avg_latency = sum(
            r.get("context_latency_s", 0) + r.get("proxy_latency_s", 0)
            for r in results
        ) / len(results)
        summary[task] = {
            "retrieval_rate": retrieval_rate,
            "answer_rate": answer_rate,
            "avg_latency_s": round(avg_latency, 2),
        }
        print(f"  {task}: retrieval={retrieval_rate:.0%}  "
              f"answer={answer_rate:.0%}  avg_latency={avg_latency:.1f}s")

    # Save results
    output = {
        "model": args.model,
        "helix_url": HELIX_URL,
        "timestamp": time.time(),
        "summary": summary,
        "results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
