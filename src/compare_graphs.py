#!/usr/bin/env python3
"""Compare two vis-network HTML graph exports and generate a merged comparison HTML.

Usage:
  python src/compare_graphs.py knowledge_graph_reuse.html knowledge_graph_noreuse.html knowledge_graph_comparison.html

Color rules:
  - Node in BOTH graphs: green
  - Node only in FIRST (reuse) graph: blue
  - Node only in SECOND (noreuse) graph: cyan

The script parses the `nodes = new vis.DataSet([...]);` and `edges = new vis.DataSet([...]);` JavaScript sections.
"""
from __future__ import annotations
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple

NODE_PATTERN = re.compile(r"nodes\s*=\s*new\s+vis\.DataSet\((\[.*?\])\);", re.DOTALL)
EDGE_PATTERN = re.compile(r"edges\s*=\s*new\s+vis\.DataSet\((\[.*?\])\);", re.DOTALL)

IN_BOTH = "#00aa00"
IN_FIRST = "#1f77ff"
IN_SECOND = "#FF7F50"
GRAY_EDGE = "#888888"


def parse_nodes_edges(html_path: Path) -> Tuple[List[Dict], List[Dict]]:
    text = html_path.read_text(encoding="utf-8")
    nodes_match = NODE_PATTERN.search(text)
    edges_match = EDGE_PATTERN.search(text)
    if not nodes_match:
        raise ValueError(f"Could not find nodes dataset in {html_path}")
    nodes_json = nodes_match.group(1)
    edges_json = edges_match.group(1) if edges_match else "[]"
    # json in file is valid; unescape sequences maintained by json.loads
    nodes = json.loads(nodes_json)
    edges = json.loads(edges_json)
    return nodes, edges


def build_comparison(first_nodes: List[Dict], second_nodes: List[Dict]) -> List[Dict]:
    first_ids = {n["id"] for n in first_nodes}
    second_ids = {n["id"] for n in second_nodes}
    all_ids = sorted(first_ids | second_ids)
    id_to_node: Dict[str, Dict] = {}

    # Prefer original label text from whichever list contains the node.
    node_lookup = {n["id"]: n for n in first_nodes}
    node_lookup.update({n["id"]: n for n in second_nodes})

    for nid in all_ids:
        base = node_lookup[nid]
        node = {
            "id": nid,
            "label": base.get("label", nid),
            "shape": base.get("shape", "dot"),
        }
        if nid in first_ids and nid in second_ids:
            node["color"] = IN_BOTH
        elif nid in first_ids:
            node["color"] = IN_FIRST
        else:
            node["color"] = IN_SECOND
        # Preserve font color behavior from originals
        node["font"] = {"color": True}
        id_to_node[nid] = node
    return list(id_to_node.values())


def merge_edges(first_edges: List[Dict], second_edges: List[Dict]) -> List[Dict]:
    # Combine edges; simple dedup by (from,to,label)
    seen = set()
    merged = []
    for edge in first_edges + second_edges:
        key = (edge.get("from"), edge.get("to"), edge.get("label"))
        if key in seen:
            continue
        seen.add(key)
        new_edge = {
            "from": edge.get("from"),
            "to": edge.get("to"),
            "label": edge.get("label"),
            "arrows": "to",
            "color": GRAY_EDGE,
            "width": 2,
        }
        merged.append(new_edge)
    return merged


def generate_html(nodes: List[Dict], edges: List[Dict], first_name: str, second_name: str) -> str:
    return f"""<html>
    <head>
        <meta charset=\"utf-8\" />
        <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css\" />
        <script src=\"https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js\"></script>
        <style>
          body {{ background-color: white; }}
          #mynetwork {{ width: 100%; height: 800px; background-color: white; border: 1px solid lightgray; }}
        </style>
    </head>
    <body>
        <h3 style=\"text-align:center;\">Graph Comparison</h3>
        <div id=\"legend\" style=\"text-align:center; margin-bottom:8px;\">
          <span style=\"color:{IN_BOTH}\">● In Both Graphs</span> &nbsp; 
          <span style=\"color:{IN_FIRST}\">● {first_name}</span> &nbsp; 
          <span style=\"color:{IN_SECOND}\">● {second_name}</span>
        </div>
        <div id=\"mynetwork\"></div>
        <script>
          var nodes = new vis.DataSet({json.dumps(nodes)});
          var edges = new vis.DataSet({json.dumps(edges)});
          var container = document.getElementById('mynetwork');
          var data = {{nodes: nodes, edges: edges}};
          var options = {{ physics: {{ enabled: true, barnesHut: {{ gravitationalConstant: -2000, centralGravity: 0.1, springLength: 200, springConstant: 0.04, damping: 0.09 }} }} }};
          var network = new vis.Network(container, data, options);
        </script>
    </body>
</html>"""


def main(argv: List[str]) -> int:
    if len(argv) != 4:
        print("Usage: python src/compare_graphs.py <first.html> <second.html> <output.html>", file=sys.stderr)
        return 1
    first = Path(argv[1])
    second = Path(argv[2])
    out = Path(argv[3])
    first_nodes, first_edges = parse_nodes_edges(first)
    second_nodes, second_edges = parse_nodes_edges(second)
    comp_nodes = build_comparison(first_nodes, second_nodes)
    comp_edges = merge_edges(first_edges, second_edges)
    html = generate_html(comp_nodes, comp_edges, first.name, second.name)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote comparison HTML to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
