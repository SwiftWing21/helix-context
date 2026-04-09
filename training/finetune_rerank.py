"""
Fine-tune DeBERTa-v3-small as a cross-encoder for gene re-ranking.

Input:  (query, gene_summary) → relevance score (0.0-1.0)
Output: Fine-tuned model saved to training/models/rerank/

Architecture: DeBERTa + regression head (MSE loss on teacher scores)

Usage:
    python training/finetune_rerank.py --data training/data/rerank_pairs.jsonl
    python training/finetune_rerank.py --data training/data/rerank_pairs.jsonl --epochs 5 --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import torch
from torch.utils.data import Dataset, DataLoader, random_split

log = logging.getLogger("helix.training.rerank")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class RerankDataset(Dataset):
    """(query, gene_summary+domains) -> relevance score."""

    def __init__(self, path: str, tokenizer, max_length: int = 256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                query = rec["query"]
                summary = rec.get("summary", "")
                domains = ", ".join(rec.get("domains", []))
                gene_text = f"{summary} [{domains}]" if domains else summary
                score = float(rec["score"])
                self.samples.append((query, gene_text, score))

        log.info("Loaded %d rerank samples from %s", len(self.samples), path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query, gene_text, score = self.samples[idx]
        encoding = self.tokenizer(
            query,
            gene_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(score, dtype=torch.float32),
        }


def train(
    data_path: str,
    model_name: str = "microsoft/deberta-v3-small",
    output_dir: str = "training/models/rerank",
    epochs: int = 3,
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
        model_name, num_labels=1, problem_type="regression",
        torch_dtype=torch.float32,
    )
    model.to(device)

    dataset = RerankDataset(data_path, tokenizer, max_length=max_length)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    log.info("Train: %d, Val: %d", train_size, val_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.squeeze(-1)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        model.train(False)
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.squeeze(-1)
                val_loss += loss_fn(preds, labels).item()

        avg_val = val_loss / max(len(val_loader), 1)
        log.info("Epoch %d/%d — train_loss=%.4f val_loss=%.4f", epoch + 1, epochs, avg_train, avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            log.info("  Saved best model (val_loss=%.4f)", avg_val)

    log.info("Training complete. Best val_loss=%.4f", best_val_loss)
    log.info("Model saved to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa for gene re-ranking")
    parser.add_argument("--data", required=True, help="Path to rerank_pairs.jsonl")
    parser.add_argument("--model", default="microsoft/deberta-v3-small", help="Base model")
    parser.add_argument("--output", default="training/models/rerank", help="Output dir")
    parser.add_argument("--epochs", type=int, default=3)
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
