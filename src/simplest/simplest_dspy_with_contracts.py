import dspy
import json
import sys
import pydantic
from pathlib import Path
from typing import Optional
from common.utils import get_lm_for_model_name, dspy_configure
from common.constants import MODEL_NAME_GEMINI_2_5_FLASH


class ContractInfo(pydantic.BaseModel):
    """Structured contract information extracted from a legal document."""
    contract_date: str = pydantic.Field(description="Date of the contract (format: YYYY-MM-DD if possible)")
    parties: str = pydantic.Field(description="Names of all parties involved in the contract")
    contract_type: str = pydantic.Field(description="Type of contract (e.g., Service Agreement, NDA, Employment)")
    subject: str = pydantic.Field(description="Main subject or purpose of the contract")
    duration: str = pydantic.Field(description="Contract duration or validity period")
    payment_terms: str = pydantic.Field(description="Payment terms if applicable")
    key_clauses: str = pydantic.Field(description="Important clauses or terms")
    signatures: str = pydantic.Field(description="Signature information")
    other_info: str = pydantic.Field(description="Any other relevant information")


class ContractExtractionSignature(dspy.Signature):
    """
    Extract all key information from a contract or legal document with 100% accuracy.
    Preserve all information exactly as it appears in the document.
    Return structured contract information including date, parties, type, subject, duration, 
    payment terms, key clauses, signatures, and other relevant details.
    """
    pdf: dspy.Image = dspy.InputField(desc="Contract or legal document PDF")
    contract_info: ContractInfo = dspy.OutputField(desc="Structured contract information")


def extract_contract_info(pdf_image: dspy.Image) -> ContractInfo:
    """Extract structured contract information from PDF."""
    extractor = dspy.Predict(ContractExtractionSignature)
    result = extractor(pdf=pdf_image)
    return result.contract_info


def generate_markdown_report(pdf_id: str, contract_info: ContractInfo) -> str:
    """Generate a markdown report for contract information."""
    md_content = f"# Contract: {pdf_id}\n\n"
    md_content += "## Contract Information\n\n"
    md_content += f"**Contract Date:** {contract_info.contract_date}\n\n"
    md_content += f"**Contract Type:** {contract_info.contract_type}\n\n"
    md_content += f"**Parties:**\n{contract_info.parties}\n\n"
    md_content += f"**Subject:**\n{contract_info.subject}\n\n"
    md_content += f"**Duration:**\n{contract_info.duration}\n\n"
    md_content += f"**Payment Terms:**\n{contract_info.payment_terms}\n\n"
    md_content += f"**Key Clauses:**\n{contract_info.key_clauses}\n\n"
    md_content += f"**Signatures:**\n{contract_info.signatures}\n\n"
    md_content += f"**Other Information:**\n{contract_info.other_info}\n\n"
    
    return md_content


def process_pdf(pdf_id: str, pdf_path: Path) -> Optional[ContractInfo]:
    """Process a single PDF and extract contract information."""
    json_output_path = pdf_path.parent / f"{pdf_id}.json"
    md_output_path = pdf_path.parent / f"{pdf_id}.md"
    
    # Skip if already processed
    if json_output_path.exists() and md_output_path.exists():
        print(f"  ⊘ Skipping {pdf_id} (already processed)")
        with open(json_output_path, 'r', encoding='utf-8') as f:
            return ContractInfo(**json.load(f))
    
    print(f"\n{'=' * 80}")
    print(f"Processing: {pdf_id}")
    print(f"{'=' * 80}\n")
    
    # Load PDF as dspy.Image
    pdf_image = dspy.Image(str(pdf_path))
    
    # Extract contract information
    print("  → Extracting contract information...")
    contract_info = extract_contract_info(pdf_image)
    
    # Generate markdown report
    md_content = generate_markdown_report(pdf_id, contract_info)
    
    # Write JSON file
    json_output_path.write_text(contract_info.model_dump_json(indent=2), encoding='utf-8')
    print(f"  ✓ JSON saved to: {json_output_path}")
    
    # Write markdown file
    md_output_path.write_text(md_content, encoding='utf-8')
    print(f"  ✓ Markdown saved to: {md_output_path}\n")
    
    return contract_info


class QuestionAnswerSignature(dspy.Signature):
    """
    Answer a question based on the extracted contract information provided.
    Use only the information from the contracts to answer the question.
    Be precise and cite relevant contract information when answering.
    """
    contracts_data: list[ContractInfo] = dspy.InputField(desc="List of all contract information")
    question: str = dspy.InputField(desc="User's question about the contracts")
    answer: str = dspy.OutputField(desc="Answer based on the contract information")


def answer_question(contracts_data: list[ContractInfo], question: str) -> str:
    """Answer a question based on extracted contract information."""
    qa_module = dspy.Predict(QuestionAnswerSignature)
    result = qa_module(contracts_data=contracts_data, question=question)
    
    return result.answer


def main():
    dspy_configure(get_lm_for_model_name(MODEL_NAME_GEMINI_2_5_FLASH, "disable", max_tokens=16384))
    
    # Get directory from command line or use default
    if len(sys.argv) > 1:
        contracts_dir = Path(sys.argv[1])
    else:
        contracts_dir = Path("src/simplest/docs/contracts")
    
    # Validate directory exists
    if not contracts_dir.exists():
        print(f"\n✗ Error: Directory does not exist: {contracts_dir}")
        print(f"\nUsage: python {sys.argv[0]} [directory_path]")
        return
    
    if not contracts_dir.is_dir():
        print(f"\n✗ Error: Path is not a directory: {contracts_dir}")
        print(f"\nUsage: python {sys.argv[0]} [directory_path]")
        return
    
    # Find all PDF files in the directory
    pdf_files = [
        f for f in contracts_dir.iterdir() 
        if f.is_file() and f.suffix.lower() == ".pdf"
    ]
    
    if not pdf_files:
        print(f"\n⚠️  No PDF files found in {contracts_dir}")
        return
    
    print(f"\n{'=' * 80}")
    print("Contract Processing Pipeline")
    print(f"{'=' * 80}")
    print(f"Directory:        {contracts_dir.absolute()}")
    print(f"PDFs found:       {len(pdf_files)}")
    print(f"{'=' * 80}\n")
    
    # Process each PDF
    contracts_data = []
    for pdf_path in sorted(pdf_files):
        pdf_id = pdf_path.stem
        try:
            contract_info = process_pdf(pdf_id, pdf_path)
            if contract_info:
                contracts_data.append(contract_info)
        except Exception as e:
            print(f"  ✗ Error processing {pdf_id}: {e}\n")
            continue
    
    print(f"\n{'=' * 80}")
    print(f"✓ Processing complete! Extracted data saved to: {contracts_dir.absolute()}")
    print(f"{'=' * 80}\n")
    
    # Interactive Q&A loop
    if contracts_data:
        print("\n" + "=" * 80)
        print("Question & Answer Mode")
        print("=" * 80)
        print("Ask questions about the contracts (or 'quit' to exit)")
        print("=" * 80 + "\n")
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                print("\n  → Processing question...")
                answer = answer_question(contracts_data, question)
                print(f"\nAnswer:\n{answer}\n")
                print("-" * 80)
                
            except KeyboardInterrupt:
                print("\n\nExiting Q&A mode...")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}\n")
                continue


if __name__ == "__main__":
    main()

