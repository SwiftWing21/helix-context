"""
Horizontal Gene Transfer -- Export and import genome files.

Usage:
    # Export current genome
    python examples/hgt_transfer.py export --description "ScoreRift knowledge"

    # Preview what an import would do
    python examples/hgt_transfer.py diff genome_export.helix

    # Import into current genome
    python examples/hgt_transfer.py import genome_export.helix

    # Import with overwrite (replace existing genes)
    python examples/hgt_transfer.py import genome_export.helix --strategy overwrite

Prerequisites:
    pip install helix-context
    A genome.db must exist (run the server or quickstart first)
"""

import argparse
import json
import sys

from helix_context.config import load_config
from helix_context.genome import Genome
from helix_context.hgt import export_genome, import_genome, genome_diff


def main():
    parser = argparse.ArgumentParser(description="Helix Horizontal Gene Transfer")
    sub = parser.add_subparsers(dest="command", required=True)

    # Export
    exp = sub.add_parser("export", help="Export genome to .helix file")
    exp.add_argument("-o", "--output", default="genome_export.helix", help="Output path")
    exp.add_argument("-d", "--description", default="", help="Description of this export")
    exp.add_argument("--include-stale", action="store_true", help="Include HETEROCHROMATIN genes")

    # Import
    imp = sub.add_parser("import", help="Import .helix file into genome")
    imp.add_argument("file", help="Path to .helix file")
    imp.add_argument("--strategy", default="skip_existing",
                     choices=["skip_existing", "overwrite", "newest"],
                     help="How to handle duplicates")

    # Diff
    dif = sub.add_parser("diff", help="Preview import without modifying genome")
    dif.add_argument("file", help="Path to .helix file")

    args = parser.parse_args()
    config = load_config()
    genome = Genome(
        path=config.genome.path,
        synonym_map=config.synonym_map,
    )

    if args.command == "export":
        result = export_genome(
            genome, args.output,
            description=args.description,
            include_heterochromatin=args.include_stale,
        )
        print(f"Exported {result['genes']} genes ({result['promoter_tags']} tags)")
        print(f"File: {result['path']} ({result['file_size']:,} bytes)")

    elif args.command == "import":
        result = import_genome(genome, args.file, merge_strategy=args.strategy)
        print(f"Imported: {result['imported']}")
        print(f"Skipped:  {result['skipped']}")
        if result['overwritten']:
            print(f"Overwritten: {result['overwritten']}")
        print(f"Source: {result['source']} ({result['source_exported_at']})")

    elif args.command == "diff":
        result = genome_diff(genome, args.file)
        print(f"Shared genes:       {result['shared']}")
        print(f"Only in file:       {result['only_in_file']} (would be imported)")
        print(f"Only in genome:     {result['only_in_genome']} (not in file)")
        print(f"Total in file:      {result['total_in_file']}")
        print(f"Total in genome:    {result['total_in_genome']}")

    genome.close()


if __name__ == "__main__":
    main()
