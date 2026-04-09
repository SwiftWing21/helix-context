"""
Fine-tune DeBERTa-v3-small for 7-class MacCartney-Manning NLI.

Input:  (gene_summary_a, gene_summary_b) -> relation class (0-6)
Output: Fine-tuned model saved to training/models/nli/

Architecture: DeBERTa + 7-class classification head (CrossEntropyLoss)

Usage:
    python training/finetune_nli.py --data training/data/nli_pairs.jsonl
    python training/finetune_nli.py --data training/data/nli_pairs.jsonl --epochs 5 --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import torch
from torch.utils.data import Dataset, DataLoader, random_split

log = logging.getLogger("helix.training.nli")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

NUM_RELATIONS = 7
RELATION_NAMES = [
    "entailment", "reverse_entailment", "equivalence",
    "alternation", "negation", "cover", "independence",
]


class NLIDataset(Dataset):
    """(text_a, text_b) -> relation class (0-6)."""

    def __init__(self, path: str, tokenizer, max_length: int = 256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                text_a = rec["text_a"]
                text_b = rec["text_b"]
                label = int(rec["label"])
                if 0 <= label < NUM_RELATIONS:
                    self.samples.append((text_a, text_b, label))

        log.info("Loaded %d NLI samples from %s", len(self.samples), path)

        # Log class distribution
        counts = [0] * NUM_RELATIONS
        for _, _, l in self.samples:
            counts[l] += 1
        for i, name in enumerate(RELATION_NAMES):
            log.info("  %s: %d (%.1f%%)", name, counts[i],
                     100 * counts[i] / max(len(self.samples), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text_a, text_b, label = self.samples[idx]
        encoding = self.tokenizer(
            text_a,
            text_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def compute_class_weights(dataset: NLIDataset, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced data."""
    counts = [0] * NUM_RELATIONS
    for _, _, l in dataset.samples:
        counts[l] += 1

    total = sum(counts)
    weights = []
    for c in counts:
        if c > 0:
            weights.append(total / (NUM_RELATIONS * c))
        else:
            weights.append(1.0)

    return torch.tensor(weights, dtype=torch.float32, device=device)


def train(
    data_path: str,
    model_name: str = "microsoft/deberta-v3-small",
    output_dir: str = "training/models/nli",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    val_split: float = 0.1,
    max_length: int = 256,
):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Loading %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_RELATIONS,
        torch_dtype=torch.float32,
    )
    model.to(device)

    dataset = NLIDataset(data_path, tokenizer, max_length=max_length)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    log.info("Train: %d, Val: %d", train_size, val_size)

    # Class-weighted loss for imbalanced data
    class_weights = compute_class_weights(dataset, device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_train = total_loss / len(train_loader)
        train_acc = correct / max(total, 1)

        # Validate
        model.train(False)
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        per_class_correct = [0] * NUM_RELATIONS
        per_class_total = [0] * NUM_RELATIONS

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                val_loss += loss_fn(outputs.logits, labels).item()

                preds = outputs.logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                for pred, true in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                    per_class_total[true] += 1
                    if pred == true:
                        per_class_correct[true] += 1

        avg_val = val_loss / max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)

        log.info(
            "Epoch %d/%d — train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f",
            epoch + 1, epochs, avg_train, train_acc, avg_val, val_acc,
        )

        # Per-class accuracy
        for i, name in enumerate(RELATION_NAMES):
            if per_class_total[i] > 0:
                cls_acc = per_class_correct[i] / per_class_total[i]
                log.info("  %s: %.3f (%d/%d)", name, cls_acc,
                         per_class_correct[i], per_class_total[i])

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            log.info("  Saved best model (val_loss=%.4f, acc=%.3f)", avg_val, val_acc)

    log.info("Training complete. Best val_loss=%.4f", best_val_loss)
    log.info("Model saved to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa for 7-class NLI")
    parser.add_argument("--data", required=True, help="Path to nli_pairs.jsonl")
    parser.add_argument("--model", default="microsoft/deberta-v3-small", help="Base model")
    parser.add_argument("--output", default="training/models/nli", help="Output dir")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
