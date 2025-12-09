from typing import List, Optional
import json
import dspy
import pydantic
import networkx as nx
from pathlib import Path
from pyvis.network import Network

from common.utils import get_lm_for_model_name, dspy_configure
from common.constants import MODEL_NAME_GEMINI_2_5_FLASH
from knowledge_graph.markdown_splitter import TextChunk, split_markdown_into_chunks
from knowledge_graph.prompts import TRIPLET_GENERAL_EXTRACTOR_INSTRUCTIONS


# 1. Define the structured output with Pydantic
class Triplet(pydantic.BaseModel):
    model_config = {"frozen": True}
    subject: str = pydantic.Field(description="The subject entity of the triplet")
    predicate: str = pydantic.Field(description="The relationship/predicate connecting subject to object")
    object: str = pydantic.Field(description="The object entity of the triplet")


class ExistingTriplets(pydantic.BaseModel):
    existing_triplets: set[Triplet] = pydantic.Field(description="List of knowledge graph triplets")

class TripletsResult(pydantic.BaseModel):
    triplets: set[Triplet] = pydantic.Field(description="List of knowledge graph triplets")


class TripletExtractionSignature(dspy.Signature):
    text: str = dspy.InputField(desc="The source text to analyze for knowledge graph triplets")
    existing_triplets: ExistingTriplets = dspy.InputField(desc="Previously extracted triplets to relate to, or empty string if none", default="")
    result: TripletsResult = dspy.OutputField(desc="A JSON object with a 'triplets' field containing a list of extracted triplets")


class TripletExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(TripletExtractionSignature.with_instructions(TRIPLET_GENERAL_EXTRACTOR_INSTRUCTIONS))

    def forward(self, text: str, existing_triplets: str = "") -> dspy.Prediction:
        return self.predictor(text=text, existing_triplets=existing_triplets)

def extract_triplets_from_text(text: str, extractor: dspy.Module, existing_triplets: Optional[List[Triplet]] = None) -> set[Triplet]:
    """Extract triplets from text using the DSPy extractor, with optional context of existing triplets."""
    result = extractor(text=text, existing_triplets=ExistingTriplets(existing_triplets=existing_triplets or []))
    return result.result.triplets


def build_networkx_graph(triplets: List[Triplet]) -> nx.DiGraph:
    """Build a NetworkX directed graph from extracted triplets."""
    G = nx.DiGraph()
    
    for triplet in triplets:
        G.add_edge(triplet.subject, triplet.object, predicate=triplet.predicate)
    
    return G


def save_triplets_as_jsonl(triplets: set[Triplet], output_file: str = "knowledge_graph_triplets.jsonl"):
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
        bgcolor="white",
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
    # dspy_configure(get_lm_for_ollama())
    dspy_configure(get_lm_for_model_name(MODEL_NAME_GEMINI_2_5_FLASH, "disable"))
    
    # Read the markdown file
    file_path = Path("src/simplest/docs/images/notes-on-linear-and-ai-agents.postprocessed.md")
    print(f"\nReading file: {file_path}")
    
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"File read successfully ({len(text)} characters)")
    
    # Split markdown into chunks
    print("\nSplitting markdown into chunks...")
    chunks: List[TextChunk] = split_markdown_into_chunks(text, strategy="headers_first")
    print(f"Split into {len(chunks)} chunks:")
    for chunk in chunks:
        header_info = f" (under: {chunk.header_context})" if chunk.header_context else ""
        print(f"  Chunk {chunk.chunk_index}: {chunk.chunk_type} ({len(chunk.content)} chars){header_info}")
    
    # Create extractor and extract triplets from each chunk for every prompt
    print(f"\n=== Extracting triplets for prompt 'GENERAL' using model: {dspy.settings.lm.model} ===")
    extractor = TripletExtractor()
    all_triplets: set[Triplet] = set()
    
    for chunk in chunks:
        print(f"\n  Processing chunk {chunk.chunk_index} ({chunk.chunk_type}, {len(chunk.content)} chars)...")
        reused_triplets = all_triplets
        if reused_triplets:
            print(f"    → Using {len(reused_triplets)} existing triplets as context")
        chunk_triplets = extract_triplets_from_text(
            chunk.content,
            extractor,
            existing_triplets=reused_triplets
        )
        print(f"    → Extracted {len(chunk_triplets)} triplets from this chunk")
        all_triplets.update(chunk_triplets)
        
        print(f"\nTotal extracted {len(all_triplets)} triplets from {len(chunks)} chunks for prompt 'GENERAL':")
        for i, triplet in enumerate(all_triplets, 1):
            print(f"  {i}. ({triplet.subject}, {triplet.predicate}, {triplet.object})")
        
        # Save triplets as JSONL
        print("\nSaving triplets as JSONL...")
        jsonl_file = "knowledge_graph_triplets.jsonl"
        save_triplets_as_jsonl(all_triplets, jsonl_file)
        
        # Build NetworkX graph
        print("\nBuilding NetworkX graph...")
        G = build_networkx_graph(all_triplets)
        print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Save graph as HTML
        print("\nSaving graph as HTML...")
        output_file = "knowledge_graph.html"
        save_graph_as_html(G, output_file)


if __name__ == "__main__":
    main()
