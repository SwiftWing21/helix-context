"""
Helix Context Quickstart — Ingest, query, and learn in 20 lines.

Prerequisites:
    pip install helix-context
    ollama pull gemma4:e2b    (or any small model)

Run:
    python examples/quickstart.py
"""

from helix_context import HelixContextManager, load_config

# Load config (uses helix.toml if present, otherwise defaults)
config = load_config()
helix = HelixContextManager(config)

# 1. Ingest some content
print("Ingesting content...")
helix.ingest("""
    Caching trades space for time. The three classical problems are:
    invalidation (when does cached data expire?), cold starts (empty cache
    after reboot), and thundering herds (many requests regenerating the
    same expired value simultaneously). TTL-based expiration is simplest;
    event-based invalidation is more precise but requires reliable events.
""")

# 2. Build context for a query
print("\nQuerying genome...")
window = helix.build_context("How does cache invalidation work?")

print(f"Genes expressed: {len(window.expressed_gene_ids)}")
print(f"Compression ratio: {window.compression_ratio:.1f}x")
print(f"Estimated tokens: {window.total_estimated_tokens}")
print(f"\nExpressed context:\n{window.expressed_context}")

# 3. Learn from an exchange (simulates post-response replication)
helix.learn(
    "How does cache invalidation work?",
    "Use TTL for simple cases, event-based invalidation for critical paths."
)

# 4. Check genome stats
stats = helix.stats()
print(f"\nGenome: {stats['total_genes']} genes, {stats['compression_ratio']:.1f}x compression")

helix.close()
print("\nDone. The genome persists in genome.db for future sessions.")
