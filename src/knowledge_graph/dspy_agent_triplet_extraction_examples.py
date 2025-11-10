from typing import Dict, List, Set, Any
import dspy
import random

# ---- start train-set -----

examples_for_triplet_extraction_train_set: List[Dict[str, Any]] = [
    # Good examples: Concise predicates, entities as objects
    {
        "text": "Linear is a communication tool. Linear provides custom instructions.",
        "expected_triplets": [
            {"subject": "Linear", "predicate": "is a", "object": "communication tool"},
            {"subject": "Linear", "predicate": "provides", "object": "custom instructions"}
        ]
    },
    {
        "text": "The product development cycle flows through discovery, planning, building, and shipping.",
        "expected_triplets": [
            {"subject": "product development cycle", "predicate": "flows through", "object": "discovery"},
            {"subject": "product development cycle", "predicate": "flows through", "object": "planning"},
            {"subject": "product development cycle", "predicate": "flows through", "object": "building"},
            {"subject": "product development cycle", "predicate": "flows through", "object": "shipping"}
        ]
    },
    {
        "text": "AI enhances each step of the product development cycle.",
        "expected_triplets": [
            {"subject": "AI", "predicate": "enhances", "object": "product development cycle"}
        ]
    },
    {
        "text": "Linear's automatic triaging system finds duplicates and provides context.",
        "expected_triplets": [
            {"subject": "Linear's automatic triaging system", "predicate": "finds", "object": "duplicates"},
            {"subject": "Linear's automatic triaging system", "predicate": "provides", "object": "context"}
        ]
    },
    {
        "text": "Humans maintain ownership of AI agents while delegating specific tasks.",
        "expected_triplets": [
            {"subject": "humans", "predicate": "maintain", "object": "ownership of AI agents"},
            {"subject": "humans", "predicate": "delegate", "object": "specific tasks"}
        ]
    },
    {
        "text": "A person owns each issue and can delegate work to coding agents.",
        "expected_triplets": [
            {"subject": "person", "predicate": "owns", "object": "issue"},
            {"subject": "person", "predicate": "delegates work to", "object": "coding agents"}
        ]
    },
    {
        "text": "Linear's Slack agent changed internal workflows by creating issues from conversations.",
        "expected_triplets": [
            {"subject": "Linear's Slack agent", "predicate": "changed", "object": "internal workflows"},
            {"subject": "Linear's Slack agent", "predicate": "creates", "object": "issues from conversations"}
        ]
    },
    {
        "text": "Linear's approach evolved from basic keyword search to semantic search to full agentic loops.",
        "expected_triplets": [
            {"subject": "Linear's approach", "predicate": "evolved from", "object": "basic keyword search"},
            {"subject": "Linear's approach", "predicate": "evolved to", "object": "semantic search"},
            {"subject": "Linear's approach", "predicate": "evolved to", "object": "full agentic loops"}
        ]
    },
    {
        "text": "Linear will orchestrate workflows where one agent delegates to other agents.",
        "expected_triplets": [
            {"subject": "Linear", "predicate": "will orchestrate", "object": "workflows"},
            {"subject": "one agent", "predicate": "delegates to", "object": "other agents"}
        ]
    },
    {
        "text": "Effective tools reflect how people actually work.",
        "expected_triplets": [
            {"subject": "effective tools", "predicate": "reflect", "object": "how people work"}
        ]
    },
    {
        "text": "Linear aims to be a natural extension of existing processes.",
        "expected_triplets": [
            {"subject": "Linear", "predicate": "aims to be", "object": "natural extension of existing processes"}
        ]
    },
    {
        "text": "Linear adapts to how you communicate.",
        "expected_triplets": [
            {"subject": "Linear", "predicate": "adapts to", "object": "how you communicate"}
        ]
    },
    # Examples addressing specific issues: Concise predicates
    {
        "text": "The system finds duplicates quickly.",
        "expected_triplets": [
            {"subject": "system", "predicate": "finds", "object": "duplicates"}
        ]
    },
    {
        "text": "The agent operates in the background.",
        "expected_triplets": [
            {"subject": "agent", "predicate": "operates in", "object": "background"}
        ]
    },
    # Examples addressing specific issues: Entities as objects (not locations)
    {
        "text": "The best AI works silently without announcing itself.",
        "expected_triplets": [
            {"subject": "best AI", "predicate": "works", "object": "silently"},
            {"subject": "best AI", "predicate": "does not announce", "object": "itself"}
        ]
    },
    {
        "text": "The triaging system works in the background.",
        "expected_triplets": [
            {"subject": "triaging system", "predicate": "works in", "object": "background"}
        ]
    },
    # Examples addressing specific issues: Correct semantic relationships
    {
        "text": "Humans maintain ownership while delegating tasks to AI agents.",
        "expected_triplets": [
            {"subject": "humans", "predicate": "maintain", "object": "ownership"},
            {"subject": "humans", "predicate": "delegate", "object": "tasks to AI agents"}
        ]
    },
    {
        "text": "People use AI agents effectively by maintaining human ownership.",
        "expected_triplets": [
            {"subject": "people", "predicate": "use", "object": "AI agents"},
            {"subject": "people", "predicate": "maintain", "object": "human ownership"}
        ]
    },
    # Examples addressing specific issues: Concrete entities as subjects
    {
        "text": "People delegate specific tasks to coding agents.",
        "expected_triplets": [
            {"subject": "people", "predicate": "delegate", "object": "specific tasks to coding agents"}
        ]
    },
    {
        "text": "Managers assign tasks to team members.",
        "expected_triplets": [
            {"subject": "managers", "predicate": "assign", "object": "tasks to team members"}
        ]
    },
    # More diverse examples
    {
        "text": "Custom instructions allow appending prompts with company-specific language and workflows.",
        "expected_triplets": [
            {"subject": "custom instructions", "predicate": "allow", "object": "appending prompts"},
            {"subject": "custom instructions", "predicate": "allow appending prompts with", "object": "company-specific language"},
            {"subject": "custom instructions", "predicate": "allow appending prompts with", "object": "workflows"}
        ]
    },
    {
        "text": "AI needs proper representation in the UI.",
        "expected_triplets": [
            {"subject": "AI", "predicate": "needs", "object": "proper representation in the UI"}
        ]
    },
    {
        "text": "Linear is building custom interfaces and concepts to make the experience more agent-native.",
        "expected_triplets": [
            {"subject": "Linear", "predicate": "is building", "object": "custom interfaces"},
            {"subject": "Linear", "predicate": "is building", "object": "concepts"},
            {"subject": "Linear", "predicate": "makes", "object": "experience more agent-native"}
        ]
    },
    {
        "text": "Automation starts with manual processes.",
        "expected_triplets": [
            {"subject": "automation", "predicate": "starts with", "object": "manual processes"}
        ]
    },
    {
        "text": "The future is about how individual agents work together.",
        "expected_triplets": [
            {"subject": "future", "predicate": "is about", "object": "how individual agents work together"}
        ]
    }
]

# ---- end train-set -----

# ---- start test-set -----

examples_for_triplet_extraction_test_set: List[Dict[str, Any]] = [
    {
        "text": "Linear started slow with AI, waiting until models were good enough to build real value.",
        "expected_triplets": [
            {"subject": "Linear", "predicate": "started slow with", "object": "AI"},
            {"subject": "Linear", "predicate": "waited until", "object": "models were good enough"},
            {"subject": "models", "predicate": "build", "object": "real value"}
        ]
    },
    {
        "text": "The best AI makes things happen under the hood without announcing itself.",
        "expected_triplets": [
            {"subject": "best AI", "predicate": "makes things happen", "object": "under the hood"},
            {"subject": "best AI", "predicate": "does not announce", "object": "itself"}
        ]
    },
    {
        "text": "Linear's automatic triaging system works in the background, taking minutes per issue to find duplicates.",
        "expected_triplets": [
            {"subject": "Linear's automatic triaging system", "predicate": "works in", "object": "background"},
            {"subject": "Linear's automatic triaging system", "predicate": "finds", "object": "duplicates"}
        ]
    },
    {
        "text": "The most effective way to use AI agents is to maintain human ownership while delegating specific tasks.",
        "expected_triplets": [
            {"subject": "most effective way to use AI agents", "predicate": "is to maintain", "object": "human ownership"},
            {"subject": "most effective way to use AI agents", "predicate": "is to delegate", "object": "specific tasks"}
        ]
    },
    {
        "text": "Linear mandates that a person owns each issue, but they can delegate work to coding agents.",
        "expected_triplets": [
            {"subject": "Linear", "predicate": "mandates", "object": "person owns each issue"},
            {"subject": "person", "predicate": "owns", "object": "issue"},
            {"subject": "person", "predicate": "can delegate", "object": "work to coding agents"}
        ]
    },
    {
        "text": "The difference between mediocre and exceptional AI is often in the prompting.",
        "expected_triplets": [
            {"subject": "difference between mediocre and exceptional AI", "predicate": "is in", "object": "prompting"}
        ]
    },
    {
        "text": "Linear provides custom instructions for all their AI tools, allowing you to append prompts with your company's specific language and workflows.",
        "expected_triplets": [
            {"subject": "Linear", "predicate": "provides", "object": "custom instructions for all their AI tools"},
            {"subject": "custom instructions", "predicate": "allow", "object": "appending prompts"},
            {"subject": "custom instructions", "predicate": "allow appending prompts with", "object": "company's specific language"},
            {"subject": "custom instructions", "predicate": "allow appending prompts with", "object": "workflows"}
        ]
    },
    {
        "text": "Linear's Slack agent changed internal workflows completely by making it effortless to create issues directly from conversations.",
        "expected_triplets": [
            {"subject": "Linear's Slack agent", "predicate": "changed", "object": "internal workflows"},
            {"subject": "Linear's Slack agent", "predicate": "creates", "object": "issues from conversations"}
        ]
    },
    {
        "text": "For AI to truly integrate into workflows, it needs proper representation in the UI.",
        "expected_triplets": [
            {"subject": "AI", "predicate": "needs", "object": "proper representation in the UI"},
            {"subject": "AI", "predicate": "integrates into", "object": "workflows"}
        ]
    },
    {
        "text": "Linear's approach evolved from basic keyword search to semantic search to full agentic loops.",
        "expected_triplets": [
            {"subject": "Linear's approach", "predicate": "evolved from", "object": "basic keyword search"},
            {"subject": "Linear's approach", "predicate": "evolved to", "object": "semantic search"},
            {"subject": "Linear's approach", "predicate": "evolved to", "object": "full agentic loops"}
        ]
    }
]

# ---- end test-set -----

def prepare_training_data(limit: int = 10000, randomize: bool = False) -> List[dspy.Example]:
    """Convert examples to DSPy format"""
    examples: List[dspy.Example] = []
    text_seen: Set[str] = set()
    
    # Add your existing examples
    combined_train_set = examples_for_triplet_extraction_train_set
    if randomize:
        random.shuffle(combined_train_set.copy())
    for i, ex in enumerate(combined_train_set):
        if i >= limit:
            break
        if ex["text"] in text_seen:
            continue
        text_seen.add(ex["text"])
        
        # Convert expected_triplets to TripletsResult format
        import pydantic
        from typing import List as ListType
        
        class Triplet(pydantic.BaseModel):
            subject: str = pydantic.Field(description="The subject entity of the triplet")
            predicate: str = pydantic.Field(description="The relationship/predicate connecting subject to object")
            object: str = pydantic.Field(description="The object entity of the triplet")
        
        class TripletsResult(pydantic.BaseModel):
            triplets: ListType[Triplet] = pydantic.Field(description="List of extracted knowledge graph triplets")
        
        triplets = [Triplet(**t) for t in ex["expected_triplets"]]
        triplets_result = TripletsResult(triplets=triplets)
        
        examples.append(dspy.Example(
            text=ex["text"],
            result=triplets_result
        ).with_inputs("text"))
    
    return examples

def prepare_test_data(limit: int = 10000, randomize: bool = False) -> List[dspy.Example]:
    """Convert test examples to DSPy format"""
    examples: List[dspy.Example] = []
    text_seen: Set[str] = set()
    
    # Add your existing examples
    combined_test_set = examples_for_triplet_extraction_test_set
    if randomize:
        random.shuffle(combined_test_set.copy())
    for i, ex in enumerate(combined_test_set):
        if i >= limit:
            break
        if ex["text"] in text_seen:
            continue
        text_seen.add(ex["text"])
        
        # Convert expected_triplets to TripletsResult format
        import pydantic
        from typing import List as ListType
        
        class Triplet(pydantic.BaseModel):
            subject: str = pydantic.Field(description="The subject entity of the triplet")
            predicate: str = pydantic.Field(description="The relationship/predicate connecting subject to object")
            object: str = pydantic.Field(description="The object entity of the triplet")
        
        class TripletsResult(pydantic.BaseModel):
            triplets: ListType[Triplet] = pydantic.Field(description="List of extracted knowledge graph triplets")
        
        triplets = [Triplet(**t) for t in ex["expected_triplets"]]
        triplets_result = TripletsResult(triplets=triplets)
        
        examples.append(dspy.Example(
            text=ex["text"],
            result=triplets_result
        ).with_inputs("text"))
    
    return examples

