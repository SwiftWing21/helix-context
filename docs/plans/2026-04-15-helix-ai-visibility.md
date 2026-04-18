# Helix AI-Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let AI agents on Helix "see" each other's work — including between their own prompts — by exposing the shipped session-registry as an AI-first surface and filling the four real gaps (inline gene attribution, unified activity feed, soft file claims, directed memo-genes).

**Architecture:** Five additive tasks on top of the existing `registry.py` / `session_registry` schema. No new storage engines — all state lands in `genome.db`. Two new REST endpoints (`/activity`, `/claims`), two schema additions (`file_claims` table, `attribution.for_handle` column), one formatting change to the expressed-context gene header, plus docs that tell agents how to use what's already there. Feature-flagged per-task via env vars so partial deployment is safe.

**Tech Stack:** Python 3.14, FastAPI, SQLite (WAL), Pydantic schemas, pytest. Matches the existing `helix-context` stack exactly — no new dependencies.

---

## Preconditions — existing infra to reuse

Before starting, confirm the following files + endpoints already exist (they do as of `helix-context v0.4.0b1`, per [SESSION_REGISTRY.md](../SESSION_REGISTRY.md)):

| Concept | Where | Status |
|---|---|---|
| `parties`, `participants`, `gene_attributions` tables | `helix_context/genome.py::_ensure_registry_schema` | shipped |
| `Registry.register_participant` / `heartbeat` / `list_participants` / `attribute_gene` | `helix_context/registry.py` | shipped |
| `GET /participants` | `helix_context/server.py:~1167` | shipped |
| `GET /authored?handle=...` | `helix_context/server.py:~1192` | shipped |
| `CWoLa` query log (`cwola.log_query`) with `session_id` + `party_id` columns | `helix_context/cwola.py` | shipped |
| Citation enrichment (`authored_by_party`, `authored_by_handle` on `/context` response) | `helix_context/server.py:~628-650` | shipped |
| `HELIX_MCP_HANDLE` / `HELIX_PARTY_ID` env contract for agent identity | `helix_context/mcp_server.py` | shipped |

**What this plan does NOT touch:**

- The expression pipeline (Steps 1-5). Retrieval quality is out of scope.
- The budget-zone spike (`budget_zone.py` — separate WIP).
- Federation to other Helix instances. Local-single-node only.

---

## File Structure

**Create:**

- `helix_context/file_claims.py` — data-access layer for soft file claims (CRUD + expiry sweep).
- `helix_context/activity_feed.py` — join-query helper for the unified queries+ingests feed.
- `tests/test_file_claims.py` — unit tests for claim CRUD + expiry.
- `tests/test_activity_feed.py` — unit tests for the feed query (seed fixtures, assert filtering).
- `tests/test_gene_header_attribution.py` — unit tests for inline attribution in expressed_context string.
- `tests/test_memo_genes.py` — unit tests for `for_handle` filter in `_express`.
- `docs/AI_VISIBILITY.md` — agent-facing guide ("how to see your teammates"). Referenced from the skill file.

**Modify:**

- `helix_context/genome.py::_ensure_registry_schema` — add `file_claims` table; add `for_handle` column to `gene_attributions`.
- `helix_context/schemas.py` — add `FileClaim` Pydantic model; extend `GeneAttribution` with optional `for_handle`.
- `helix_context/registry.py` — extend `attribute_gene()` to accept `for_handle`; add `get_memos_for_handle()` helper.
- `helix_context/context_manager.py::_express` (or the gene-header formatter called by `build_context`) — (a) prepend unexpired memos addressed to caller; (b) inline `authored_by=<handle>@<ts>` into each `<GENE ...>` header when attribution exists.
- `helix_context/server.py` — new `POST /claim` / `GET /claims` / `DELETE /claim` endpoints; new `GET /activity` endpoint; thread `agent_handle` capture through `/context` and `/ingest` if not already (likely already is).
- `helix_context/mcp_server.py` — expose the new endpoints as MCP tools (`helix_activity`, `helix_claim_file`).

---

## Task 0 — Baseline verification (no code)

**Goal:** Confirm the "shipped" preconditions above actually exist on the current branch before writing new code.

- [ ] **Step 1: Read and skim `registry.py`**

  Confirm `register_participant`, `heartbeat`, `list_participants`, `attribute_gene` all exist and match the signatures in SESSION_REGISTRY.md.

- [ ] **Step 2: Curl `GET /participants`**

  Expected: non-empty JSON array when at least one MCP client is running, each entry with `handle`, `party_id`, `status`, `last_seen`, `workspace` fields.

  ```bash
  curl -s http://127.0.0.1:11437/participants | python -m json.tool
  ```

- [ ] **Step 3: Curl `GET /authored?handle=raude`**

  Expected: list of gene summaries with `gene_id`, `authored_at`, `preview`, recency-sorted.

  ```bash
  curl -s "http://127.0.0.1:11437/authored?handle=raude" | python -m json.tool
  ```

- [ ] **Step 4: Read `context_manager._express` and find where the `<GENE src="..." facts="...">` header string is built.**

  Note the line number. Task 1 modifies exactly that site.

- [ ] **Step 5: Read `genome.py::_ensure_registry_schema`.**

  Note the current DDL — Task 4 adds `file_claims`, Task 5 adds `for_handle`.

- [ ] **No commit** (read-only step).

---

## Task 1 — Inline attribution in expressed gene headers

**Goal:** Show `authored_by=<handle>@<ISO-ts>` directly in each `<GENE ...>` header in `expressed_context`, so an AI agent sees "raude wrote this gene 10 minutes ago" without parsing a separate citations field.

**Why this is Task 1:** It's the smallest, touches one formatter, and no new endpoints. Warms up the codebase before the schema-change tasks.

**Files:**
- Modify: `helix_context/context_manager.py` — the site identified in Task 0 Step 4.
- Modify: `helix_context/server.py` — ensure the citation-enrichment query already fired runs before the header formatter (it should — Task 0 confirms order).
- Test: `tests/test_gene_header_attribution.py`

- [ ] **Step 1: Write the failing test**

  ```python
  # tests/test_gene_header_attribution.py
  from helix_context.context_manager import format_gene_header  # create if not exported

  def test_header_includes_authored_by_when_attribution_present():
      gene = _stub_gene(
          gene_id="abc123",
          src="helix_context/budget_zone.py",
          facts="window=128000 zones=5",
      )
      attribution = {"handle": "raude", "authored_at": 1712500000}
      header = format_gene_header(gene, attribution=attribution)
      assert 'authored_by="raude@2026-04-07T' in header
      assert 'src="helix_context/budget_zone.py"' in header

  def test_header_omits_authored_by_when_no_attribution():
      gene = _stub_gene(gene_id="abc123", src="foo.py", facts="x=1")
      header = format_gene_header(gene, attribution=None)
      assert "authored_by" not in header
  ```

- [ ] **Step 2: Run test to verify it fails**

  ```bash
  pytest tests/test_gene_header_attribution.py -v
  ```
  Expected: ImportError on `format_gene_header` (if we extract it) OR missing attribute assertion.

- [ ] **Step 3: Refactor the header-formatting code into `format_gene_header(gene, attribution=None)` if it's currently inline.**

  Keep behavior identical when `attribution is None`. Add:
  ```python
  if attribution and attribution.get("handle"):
      ts = attribution.get("authored_at")
      ts_iso = datetime.utcfromtimestamp(ts).isoformat() + "Z" if ts else "?"
      attrs.append(f'authored_by="{attribution["handle"]}@{ts_iso}"')
  ```

- [ ] **Step 4: Wire the citation-enrichment dict that already runs in `server.py:~628-650` into `build_context`'s header formatter.**

  The enrichment currently fires AFTER header formatting and attaches to citations. Move the attribution-lookup one step earlier (or pass through) so headers can use it. The query is batched, so performance impact is ~0.

- [ ] **Step 5: Run the test to verify it passes**

  ```bash
  pytest tests/test_gene_header_attribution.py -v
  ```

- [ ] **Step 6: Integration smoke test**

  ```bash
  # Start server, have raude ingest one attributed gene, query /context, confirm header includes authored_by
  curl -s -X POST http://127.0.0.1:11437/ingest -H "Content-Type: application/json" \
    -d '{"content":"test fact: port is 9999","content_type":"text","agent_handle":"raude"}'
  curl -s -X POST http://127.0.0.1:11437/context -H "Content-Type: application/json" \
    -d '{"query":"what port"}' | grep -o 'authored_by="[^"]*"' | head
  ```

- [ ] **Step 7: Commit**

  ```bash
  git add tests/test_gene_header_attribution.py helix_context/context_manager.py helix_context/server.py
  git commit -m "feat(registry): inline authored_by in expressed-context gene headers"
  ```

---

## Task 2 — Unified activity feed: `GET /activity?since=<ts>`

**Goal:** One endpoint returns all `/context` queries + `/ingest` ingests by OTHER agents since a timestamp. Solves the "what did the team do while I was idle" gap. This is the headline "between-prompts" feature.

**Why this task:** It's pure read-side plumbing — the raw data (CWoLa query log, gene_attributions) already exists.

**Files:**
- Create: `helix_context/activity_feed.py`
- Create: `tests/test_activity_feed.py`
- Modify: `helix_context/server.py` — new `GET /activity` endpoint.
- Modify: `helix_context/mcp_server.py` — expose as `helix_activity` MCP tool.

- [ ] **Step 1: Write the failing test**

  ```python
  # tests/test_activity_feed.py
  def test_activity_returns_queries_and_ingests_since_ts(seeded_genome):
      # seeded_genome fixture inserts:
      #   - query by raude at t=100
      #   - ingest by laude at t=150
      #   - query by taude at t=200 (the caller — should be excluded)
      feed = get_activity(seeded_genome, since=50, caller_handle="taude")
      assert len(feed["queries"]) == 1
      assert feed["queries"][0]["handle"] == "raude"
      assert feed["queries"][0]["ts"] == 100
      assert len(feed["ingests"]) == 1
      assert feed["ingests"][0]["handle"] == "laude"

  def test_activity_excludes_self(seeded_genome):
      feed = get_activity(seeded_genome, since=0, caller_handle="raude")
      assert all(q["handle"] != "raude" for q in feed["queries"])
      assert all(i["handle"] != "raude" for i in feed["ingests"])

  def test_activity_honors_since_filter(seeded_genome):
      feed = get_activity(seeded_genome, since=180, caller_handle="taude")
      assert feed["queries"] == []  # raude at 100 is before 180
      assert feed["ingests"] == []  # laude at 150 is before 180
  ```

- [ ] **Step 2: Run test to verify it fails**

  ```bash
  pytest tests/test_activity_feed.py -v
  ```
  Expected: ImportError on `get_activity`.

- [ ] **Step 3: Implement `activity_feed.get_activity(genome, since, caller_handle, limit=100)`**

  Two SELECTs:
  ```python
  def get_activity(genome, since: float, caller_handle: str, limit: int = 100) -> dict:
      cur = genome.conn.cursor()
      # Queries from CWoLa log, joined to participants for handle
      queries = cur.execute("""
        SELECT q.ts, p.handle, q.query_text, q.session_id
        FROM cwola_query_log q
        LEFT JOIN participants p ON p.party_id = q.party_id
        WHERE q.ts >= ?
          AND COALESCE(p.handle, '') != ?
        ORDER BY q.ts DESC LIMIT ?
      """, (since, caller_handle, limit)).fetchall()

      # Ingests from gene_attributions
      ingests = cur.execute("""
        SELECT a.authored_at AS ts, a.handle, g.source_id AS src, g.gene_id
        FROM gene_attributions a
        JOIN genes g ON g.gene_id = a.gene_id
        WHERE a.authored_at >= ? AND a.handle != ?
        ORDER BY a.authored_at DESC LIMIT ?
      """, (since, caller_handle, limit)).fetchall()

      return {
          "queries": [dict(q) for q in queries],
          "ingests": [dict(i) for i in ingests],
          "since": since,
          "caller_handle": caller_handle,
      }
  ```

- [ ] **Step 4: Run test to verify it passes**

  ```bash
  pytest tests/test_activity_feed.py -v
  ```

- [ ] **Step 5: Add the endpoint**

  ```python
  # server.py
  @app.get("/activity")
  async def activity(since: float, handle: str = "", limit: int = 100):
      from .activity_feed import get_activity
      return get_activity(helix.genome, since=since, caller_handle=handle, limit=limit)
  ```

- [ ] **Step 6: Integration test**

  ```bash
  curl -s "http://127.0.0.1:11437/activity?since=0&handle=taude&limit=10" | python -m json.tool
  ```
  Expected: `{queries: [...], ingests: [...], since: 0, caller_handle: "taude"}`.

- [ ] **Step 7: Add MCP tool**

  In `mcp_server.py`, register a `helix_activity` tool that calls the endpoint and returns a human-readable summary ("While you were idle: raude queried X, laude ingested Y").

- [ ] **Step 8: Commit**

  ```bash
  git add helix_context/activity_feed.py tests/test_activity_feed.py helix_context/server.py helix_context/mcp_server.py
  git commit -m "feat(registry): GET /activity — unified queries+ingests feed for cross-agent visibility"
  ```

---

## Task 3 — Polish `/participants` response for AI consumers

**Goal:** Ensure `list_participants` returns the fields an AI needs to make decisions (`last_seen`, `current_cwd`, `last_query_preview`, `last_ingested_src`), and add a `GET /agents` alias so the URL reads like it behaves. Document the endpoint in a new `docs/AI_VISIBILITY.md`.

**Why after Task 2:** Task 2 proves the shape of what AI agents want; Task 3 applies that shape to the existing presence endpoint.

**Files:**
- Modify: `helix_context/registry.py::list_participants` — add the missing fields if absent.
- Modify: `helix_context/server.py` — add `GET /agents` alias pointing at `list_participants`.
- Create: `docs/AI_VISIBILITY.md` — agent-facing guide covering `/agents`, `/activity`, `/authored`, memos, claims.
- Test: extend `tests/test_registry.py`.

- [ ] **Step 1: Inspect `Participant` schema and `list_participants` output**

  Confirm whether `last_query_preview` and `last_ingested_src` exist. If yes, skip Step 2.

- [ ] **Step 2 (conditional): Add the missing fields**

  Extend `schemas.Participant` with `last_query_preview: Optional[str]` and `last_ingested_src: Optional[str]`. Populate in `register_participant` / `heartbeat` update paths. Write a test first that asserts the fields appear in the response.

- [ ] **Step 3: Add `GET /agents` alias**

  ```python
  @app.get("/agents")
  async def agents_alias(party_id: str = "", status: str = "", limit: int = 50):
      return await participants_endpoint(party_id=party_id, status=status, limit=limit)
  ```

- [ ] **Step 4: Write `docs/AI_VISIBILITY.md`**

  Cover:
  - "Who else is working?" → `GET /agents`
  - "What happened while I was idle?" → `GET /activity?since=<last_turn_ts>`
  - "What has raude written recently?" → `GET /authored?handle=raude`
  - "Claim a file" → `POST /claim` (from Task 4)
  - "Leave a memo for another agent" → `/ingest` with `content_type=memo, for_handle=...` (from Task 5)

  Also explain the handle contract: `HELIX_MCP_HANDLE` env var at MCP-server start, or `agent_handle` field in request body for raw HTTP.

- [ ] **Step 5: Commit**

  ```bash
  git add helix_context/registry.py helix_context/schemas.py helix_context/server.py \
          docs/AI_VISIBILITY.md tests/test_registry.py
  git commit -m "feat(registry): GET /agents alias + last_query/last_ingest fields + AI_VISIBILITY.md"
  ```

---

## Task 4 — Soft file claims

**Goal:** A best-effort lock so two agents don't step on each other when editing the same file. Not an enforced lock — just a discoverable claim with a TTL.

**Files:**
- Create: `helix_context/file_claims.py`
- Create: `tests/test_file_claims.py`
- Modify: `helix_context/genome.py::_ensure_registry_schema` — add `file_claims` table.
- Modify: `helix_context/server.py` — `POST /claim`, `GET /claims`, `DELETE /claim`.
- Modify: `helix_context/server.py::context_endpoint` — include overlapping claims in the response when `session_context.active_files` is supplied.
- Modify: `helix_context/mcp_server.py` — expose `helix_claim_file` / `helix_release_file` tools.

- [ ] **Step 1: Write failing test for claim CRUD**

  ```python
  # tests/test_file_claims.py
  def test_create_claim_persists(genome):
      claim = create_claim(genome, handle="taude",
                           file_path="helix_context/context_manager.py",
                           ttl_s=1800)
      assert claim["expires_at"] > time.time() + 1700

  def test_get_overlapping_claims_returns_others_only(genome):
      create_claim(genome, handle="raude", file_path="a.py", ttl_s=1800)
      create_claim(genome, handle="taude", file_path="b.py", ttl_s=1800)
      overlap = get_overlapping_claims(genome, caller_handle="taude",
                                       files=["a.py", "b.py"])
      assert len(overlap) == 1
      assert overlap[0]["handle"] == "raude"
      assert overlap[0]["file_path"] == "a.py"

  def test_expired_claims_are_filtered(genome):
      create_claim(genome, handle="raude", file_path="a.py", ttl_s=-10)  # already expired
      overlap = get_overlapping_claims(genome, caller_handle="taude", files=["a.py"])
      assert overlap == []
  ```

- [ ] **Step 2: Add schema**

  ```python
  # genome.py::_ensure_registry_schema — append
  cur.execute("""
    CREATE TABLE IF NOT EXISTS file_claims (
      claim_id TEXT PRIMARY KEY,
      handle TEXT NOT NULL,
      party_id TEXT,
      file_path TEXT NOT NULL,
      claimed_at REAL NOT NULL,
      expires_at REAL NOT NULL,
      note TEXT
    )
  """)
  cur.execute("CREATE INDEX IF NOT EXISTS idx_file_claims_file_exp "
              "ON file_claims(file_path, expires_at)")
  cur.execute("CREATE INDEX IF NOT EXISTS idx_file_claims_handle "
              "ON file_claims(handle)")
  ```

- [ ] **Step 3: Implement `file_claims.py`**

  Functions: `create_claim(genome, handle, file_path, ttl_s, note=None)`, `release_claim(genome, claim_id)`, `get_overlapping_claims(genome, caller_handle, files)`, `sweep_expired(genome)`. Read-side filters `WHERE expires_at > strftime('%s', 'now')`.

- [ ] **Step 4: Run tests; pass**

  ```bash
  pytest tests/test_file_claims.py -v
  ```

- [ ] **Step 5: Add endpoints**

  ```python
  @app.post("/claim")
  async def claim_file(req: ClaimRequest):
      return file_claims.create_claim(helix.genome, **req.dict())

  @app.get("/claims")
  async def list_claims(handle: str = "", file: str = ""): ...

  @app.delete("/claim/{claim_id}")
  async def release(claim_id: str): ...
  ```

- [ ] **Step 6: Surface in `/context` response when relevant**

  In `context_endpoint`, if `session_context.active_files` is supplied, call `get_overlapping_claims` and add a top-level field to the response:
  ```python
  response["foreign_claims"] = overlapping_claims  # list, often empty
  ```
  AI callers see "raude is in context_manager.py, heads up" before they start editing.

- [ ] **Step 7: Register background expiry sweep**

  Add `file_claims.sweep_expired` to the existing `_background_registry_sweep` coroutine in server.py. Interval: 60s. Cheap DELETE on indexed column.

- [ ] **Step 8: Commit**

  ```bash
  git add helix_context/file_claims.py tests/test_file_claims.py \
          helix_context/genome.py helix_context/server.py helix_context/mcp_server.py
  git commit -m "feat(registry): soft file claims — POST /claim + overlap warnings on /context"
  ```

---

## Task 5 — Directed memo-genes (`for_handle`)

**Goal:** Let an agent drop a note addressed to a specific sibling (`for_handle="taude"`). On that sibling's next `/context` call, any unexpired memos for them are prepended to the expressed context regardless of topic match.

**Files:**
- Modify: `helix_context/genome.py::_ensure_registry_schema` — add `for_handle TEXT` column to `gene_attributions` (ALTER TABLE).
- Modify: `helix_context/schemas.py::GeneAttribution` — add optional `for_handle: Optional[str]`.
- Modify: `helix_context/registry.py::attribute_gene` — accept and store `for_handle`. Add `get_memos_for_handle(handle, since=0, limit=5)`.
- Modify: `helix_context/server.py::ingest_endpoint` — pass through `for_handle` body field.
- Modify: `helix_context/context_manager.py::_express` — pre-pass that prepends unexpired memos addressed to the caller.
- Test: `tests/test_memo_genes.py`

- [ ] **Step 1: Write failing test**

  ```python
  # tests/test_memo_genes.py
  def test_memo_is_expressed_even_on_off_topic_query(client, genome):
      # raude ingests a memo for taude
      client.post("/ingest", json={
        "content": "Sprint 3 gate waiting on A — see CWoLa sweep",
        "content_type": "memo",
        "agent_handle": "raude",
        "for_handle": "taude",
      })
      # taude queries something unrelated
      r = client.post("/context", json={
        "query": "what color is the sky",
        "agent_handle": "taude",
      })
      body = r.json()[0]["content"]
      assert "Sprint 3 gate" in body  # memo surfaced despite topic mismatch

  def test_memo_not_surfaced_to_other_handles(client, genome):
      client.post("/ingest", json={
        "content": "for taude only", "content_type": "memo",
        "agent_handle": "raude", "for_handle": "taude",
      })
      r = client.post("/context", json={
        "query": "anything", "agent_handle": "laude",
      })
      assert "for taude only" not in r.json()[0]["content"]

  def test_memo_consumed_or_decays(client, genome):
      # Mark-as-read via a second /context call from taude with ack=True
      client.post("/ingest", json={"content": "memo", "content_type": "memo",
                                   "agent_handle": "raude", "for_handle": "taude"})
      r1 = client.post("/context", json={"query": "x", "agent_handle": "taude", "ack_memos": True})
      assert "memo" in r1.json()[0]["content"]
      r2 = client.post("/context", json={"query": "x", "agent_handle": "taude"})
      assert "memo" not in r2.json()[0]["content"]
  ```

- [ ] **Step 2: Schema migration**

  ```python
  # genome.py
  cur.execute("ALTER TABLE gene_attributions ADD COLUMN for_handle TEXT")
  # wrap in try/except sqlite3.OperationalError for idempotency (existing dbs)
  ```

- [ ] **Step 3: Update `attribute_gene` + schema + registry helper**

  ```python
  def get_memos_for_handle(self, handle: str, since: float = 0, limit: int = 5) -> List[Tuple[Gene, float]]:
      cur = self._conn.cursor()
      rows = cur.execute("""
        SELECT g.*, a.authored_at FROM genes g
        JOIN gene_attributions a ON a.gene_id = g.gene_id
        WHERE a.for_handle = ? AND a.authored_at >= ?
          AND a.acked_at IS NULL
        ORDER BY a.authored_at DESC LIMIT ?
      """, (handle, since, limit)).fetchall()
      return [(self._row_to_gene(r), r["authored_at"]) for r in rows]
  ```

  Add `acked_at REAL` column alongside `for_handle` — memos are once-deliverable unless the caller opts-out of ack.

- [ ] **Step 4: Plumb through `/ingest`**

  Accept `for_handle` in body. When present, pass to `attribute_gene`. Coerce `content_type` to `"memo"` if `for_handle` set without explicit type (backward-safe).

- [ ] **Step 5: Plumb through `_express`**

  At the start of `_express` (before promoter extraction), if caller_handle is known:
  ```python
  memos = self.registry.get_memos_for_handle(caller_handle, since=time.time() - 86400)
  if memos:
      # Prepend to candidates; these bypass the tier/cap logic
      memo_genes = [g for g, _ts in memos]
      candidates = memo_genes + candidates
      if ack_memos:
          self.registry.ack_memos([g.gene_id for g in memo_genes])
  ```

- [ ] **Step 6: Run tests**

  ```bash
  pytest tests/test_memo_genes.py -v
  ```

- [ ] **Step 7: Manual smoke test**

  ```bash
  curl -s -X POST http://127.0.0.1:11437/ingest -H "Content-Type: application/json" \
    -d '{"content":"hey taude, check the bench","content_type":"memo","agent_handle":"raude","for_handle":"taude"}'
  curl -s -X POST http://127.0.0.1:11437/context -H "Content-Type: application/json" \
    -d '{"query":"unrelated topic","agent_handle":"taude","ack_memos":true}' | \
    python -c "import sys,json; print(json.load(sys.stdin)[0]['content'][:500])"
  ```
  Expected: memo content appears near the top.

- [ ] **Step 8: Document the channel in `docs/AI_VISIBILITY.md`** (extends Task 3).

- [ ] **Step 9: Commit**

  ```bash
  git add helix_context/genome.py helix_context/schemas.py helix_context/registry.py \
          helix_context/server.py helix_context/context_manager.py \
          tests/test_memo_genes.py docs/AI_VISIBILITY.md
  git commit -m "feat(registry): memo-genes — for_handle attribution + preferred expression for addressee"
  ```

---

## Ordering rationale

1 → 2 → 3 → 4 → 5 is ordered by (a) risk (Task 1 is single-formatter, Task 5 touches the hot path in `_express`), (b) infra dependency (Tasks 2-3 prepare the reader-side model the later tasks rely on), and (c) demo value (Task 2 alone produces a usable agent surface; the rest are multipliers).

Each task is independently shippable. Partial deployment is safe — if Task 5 turns out to need redesign, Tasks 1-4 stand alone.

## Env-flag rollout

None of these need feature flags. They are all additive: new endpoints, new fields, new tables. The one exception is **Task 5's `_express` change** — prepending memos alters the expression pipeline. Guard it with `HELIX_MEMO_GENES=1` at first; remove the flag after one week of green runs.

## Test commands cheatsheet

```bash
# Full suite
pytest tests/ -v

# Just the new tests
pytest tests/test_gene_header_attribution.py \
       tests/test_activity_feed.py \
       tests/test_file_claims.py \
       tests/test_memo_genes.py -v

# Live integration (requires server running)
pytest tests/ -m live -v -s
```

## Open questions to resolve at execution time

1. **Memo TTL.** 24h default above. Could be shorter (1h) if we expect agent turn-around times of minutes. Revisit after Week 1 of usage.
2. **Claim TTL.** Default 30min. Should a claim auto-extend if the claimant keeps editing that file? Probably yes — add a `/claim/{id}/refresh` endpoint, skip for now.
3. **Activity feed `since` default.** Endpoint requires `since` — should it default to "last participant heartbeat for caller" if omitted? Nice-to-have, not blocking.
4. **Self-filtering basis.** `handle` comparison is exact-string. If an agent restarts and gets a new `participant_id` but same handle, filtering still works. If they change handles, they'll see their own activity. Acceptable.

## Post-merge: update the client skill

After all 5 tasks ship, edit `~/.claude/skills/helix-context/SKILL.md` to add an "Ambient awareness" section covering `/agents`, `/activity`, claims, and memos — so future Claude instances find these naturally.
