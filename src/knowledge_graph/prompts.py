from typing import Dict

TRIPLET_GENERAL_EXTRACTOR_INSTRUCTIONS = """
Extract concrete (not abstract) knowledge graph triplets (subject, predicate, object) from the input text.
If existing_triplets are provided, refer to them and avoid duplicates.
"""

TRIPLET_TECH_RELATIONS_INSTRUCTIONS = """
Extract knowledge graph triplets that describe relationships between technologies (software, tools, frameworks, hardware, protocols, etc.). Focus on how one technology depends on, competes with, extends, or integrates with another technology. Ignore organizational or non-technical entities.
"""

TRIPLET_COMPANY_RELATIONS_INSTRUCTIONS = """
Extract knowledge graph triplets that describe relationships between companies or organizations (partnerships, acquisitions, competition, supplier/customer relations, joint ventures, etc.). Ignore purely technological relationships unless they directly describe an interaction between companies.
"""

PROMPTS: Dict[str, str] = {
    "GENERAL": TRIPLET_GENERAL_EXTRACTOR_INSTRUCTIONS,
    "TECH_RELATIONS": TRIPLET_TECH_RELATIONS_INSTRUCTIONS,
    "COMPANY_RELATIONS": TRIPLET_COMPANY_RELATIONS_INSTRUCTIONS,
}