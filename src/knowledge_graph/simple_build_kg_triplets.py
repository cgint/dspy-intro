from typing import List
import json
import dspy
import pydantic
import networkx as nx
from pathlib import Path
from pyvis.network import Network

from common.utils import get_lm_for_model_name, dspy_configure
from common.constants import MODEL_NAME_GEMINI_2_5_FLASH


# 1. Define the structured output with Pydantic
class Triplet(pydantic.BaseModel):
    subject: str = pydantic.Field(description="The subject entity of the triplet")
    predicate: str = pydantic.Field(description="The relationship/predicate connecting subject to object")
    object: str = pydantic.Field(description="The object entity of the triplet")


class TripletsResult(pydantic.BaseModel):
    triplets: List[Triplet] = pydantic.Field(description="List of extracted knowledge graph triplets")


# 2. Define the DSPy Signature
class TripletExtractionSignature(dspy.Signature):
    """
    Extract knowledge graph triplets (subject-predicate-object) from the given text.
    Each triplet represents a meaningful relationship between entities.
    Extract all significant relationships, concepts, and connections mentioned in the text.
    Focus on concrete relationships rather than abstract concepts.
    Return the result as a JSON object with a "triplets" field containing a list of triplets.
    Each triplet should have "subject", "predicate", and "object" fields.
    """

    text: str = dspy.InputField(desc="The source text to analyze for knowledge graph triplets")
    result: TripletsResult = dspy.OutputField(desc="A JSON object with a 'triplets' field containing a list of extracted triplets")


# 3. Create the DSPy Module
class TripletExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(TripletExtractionSignature)

    def forward(self, text: str) -> dspy.Prediction:
        return self.predictor(text=text)


def extract_triplets_from_text(text: str, extractor: TripletExtractor) -> List[Triplet]:
    """Extract triplets from text using the DSPy extractor."""
    result = extractor(text=text)
    return result.result.triplets


def build_networkx_graph(triplets: List[Triplet]) -> nx.DiGraph:
    """Build a NetworkX directed graph from extracted triplets."""
    G = nx.DiGraph()
    
    for triplet in triplets:
        G.add_edge(triplet.subject, triplet.object, predicate=triplet.predicate)
    
    return G


def save_triplets_as_jsonl(triplets: List[Triplet], output_file: str = "knowledge_graph_triplets.jsonl"):
    """Save triplets as a JSONL file (one JSON object per line)."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for triplet in triplets:
            json_line = json.dumps({
                "subject": triplet.subject,
                "predicate": triplet.predicate,
                "object": triplet.object
            }, ensure_ascii=False)
            f.write(json_line + '\n')
    print(f"Triplets saved to {output_file}")


def save_graph_as_html(G: nx.DiGraph, output_file: str = "knowledge_graph.html"):
    """Save the NetworkX graph as an interactive HTML file using pyvis."""
    # Create a pyvis network
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#222222",
        font_color=True,
        directed=True,
        notebook=False
    )
    
    # Set physics options for better layout
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.04,
          "damping": 0.09
        }
      }
    }
    """)
    
    # Add nodes and edges from NetworkX graph
    for node in G.nodes():
        net.add_node(node, label=node, color="#97c2fc", font={"size": 14})
    
    for u, v, data in G.edges(data=True):
        predicate = data.get('predicate', '')
        net.add_edge(u, v, label=predicate, color="#888888", width=2, arrows="to")
    
    # Save to HTML file
    net.save_graph(output_file)
    print(f"\nGraph saved to {output_file}")
    print("You can open it in your browser to view the interactive graph.")


def main():
    # Configure DSPy
    dspy_configure(get_lm_for_model_name(MODEL_NAME_GEMINI_2_5_FLASH, "disable"))
    
    # Read the markdown file
    file_path = Path("src/simplest/docs/images/notes-on-linear-and-ai-agents.md")
    print(f"\nReading file: {file_path}")
    
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"File read successfully ({len(text)} characters)")
    
    # Create extractor and extract triplets
    print("\nExtracting triplets using DSPy...")
    extractor = TripletExtractor()
    triplets = extract_triplets_from_text(text, extractor)
    
    print(f"\nExtracted {len(triplets)} triplets:")
    for i, triplet in enumerate(triplets, 1):
        print(f"  {i}. ({triplet.subject}, {triplet.predicate}, {triplet.object})")
    
    # Save triplets as JSONL
    print("\nSaving triplets as JSONL...")
    jsonl_file = "knowledge_graph_triplets.jsonl"
    save_triplets_as_jsonl(triplets, jsonl_file)
    
    # Build NetworkX graph
    print("\nBuilding NetworkX graph...")
    G = build_networkx_graph(triplets)
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Save graph as HTML
    print("\nSaving graph as HTML...")
    output_file = "knowledge_graph.html"
    save_graph_as_html(G, output_file)


if __name__ == "__main__":
    main()
