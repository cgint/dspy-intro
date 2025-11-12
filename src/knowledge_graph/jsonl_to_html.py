"""Utility to build an interactive HTML knowledge graph from a JSONL triplets file.

Each line of the JSONL file must be a JSON object with keys: subject, predicate, object.

Provides a single function `create_html_from_triplets(jsonl_path, output_html)` that can be
imported externally or invoked via CLI.

Example CLI usage:
    python src/knowledge_graph/jsonl_to_html.py --input knowledge_graph_triplets_g_filtered.jsonl \
        --output knowledge_graph_filtered.html
"""
from __future__ import annotations

from typing import List
from pathlib import Path
import json
import argparse

from knowledge_graph.simple_build_kg_triplets import (
    Triplet,
    build_networkx_graph,
    save_graph_as_html,
)


def load_triplets_from_jsonl(jsonl_path: str | Path) -> List[Triplet]:
    """Load triplets from a JSONL file into Triplet objects."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Triplets file not found: {path}")
    triplets: List[Triplet] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Tolerate keys that may be differently cased
            triplets.append(Triplet(
                subject=data["subject"],
                predicate=data["predicate"],
                object=data["object"],
            ))
    return triplets


def create_html_from_triplets(jsonl_path: str | Path, output_html: str | Path = "knowledge_graph_from_jsonl.html") -> Path:
    """Create an interactive HTML knowledge graph from a JSONL triplets file.

    Returns the path to the created HTML file.
    """
    triplets = load_triplets_from_jsonl(jsonl_path)
    G = build_networkx_graph(triplets)
    output_html = Path(output_html)
    save_graph_as_html(G, str(output_html))
    return output_html


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HTML knowledge graph from JSONL triplets")
    parser.add_argument("--input", "-i", required=True, help="Path to JSONL triplets file")
    parser.add_argument("--output", "-o", default="knowledge_graph_from_jsonl.html", help="Output HTML file path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    html_path = create_html_from_triplets(args.input, args.output)
    print(f"HTML graph written to: {html_path}")


if __name__ == "__main__":
    main()
