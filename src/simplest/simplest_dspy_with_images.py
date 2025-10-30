import dspy
from pathlib import Path
from PIL import Image
from common.utils import get_lm_for_model_name, dspy_configure
from common.constants import MODEL_NAME_GEMINI_2_5_FLASH


class ImageTranscriptionSignature(dspy.Signature):
    """
    Make an exact transcript of all text on the image. 
    The image may contain both English and German text.
    Preserve all text exactly as written in both languages.
    In case you need to do corrections, only do so if it is completely clear from the context of the identified text. Otherwise, leave the text as is.

    Return the text representation of the image as structured markdown, also use formatting elements like heading, bold, italic, underline, bullet points, tables, etc. to highlight important text.
    """
    image: dspy.Image = dspy.InputField()
    transcription: str = dspy.OutputField(desc="The text representation of the image as structured markdown.")


def image_transcriber(image: dspy.Image) -> str:
    transcriber = dspy.Predict(ImageTranscriptionSignature)
    transcription_response = transcriber(image=image)
    return transcription_response.transcription


def generate_markdown_report(image_path: Path, transcription: str) -> str:
    """Generate a markdown report for an image transcription."""
    md_content = f"# Transcription: {image_path.name}\n\n"
    md_content += f"**Source Image:** `{image_path.name}`\n\n"
    md_content += "---\n\n"
    md_content += f"{transcription}\n"
    
    return md_content


def process_image(image_path: Path, output_dir: Path):
    """Process a single image and generate a markdown report."""
    print(f"\n{'=' * 80}")
    print(f"Processing: {image_path.name}")
    print(f"{'=' * 80}\n")
    
    # Load image
    with Image.open(image_path) as pil_image:
        image = dspy.Image(pil_image)
        
        # Transcribe text
        print("  → Transcribing text from image...")
        transcription = image_transcriber(image)
        
        # Generate markdown report
        md_content = generate_markdown_report(image_path, transcription)
        
        # Write to file
        output_path = output_dir / f"{image_path.stem}.md"
        output_path.write_text(md_content)
        print(f"  ✓ Transcription saved to: {output_path}\n")


def main():
    dspy_configure(get_lm_for_model_name(MODEL_NAME_GEMINI_2_5_FLASH, "disable"))
    
    # Configuration
    input_dir = Path("src/simplest/docs/images")
    output_dir = Path("src/simplest/docs/images")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    
    # Find all images in the directory
    image_files = [
        f for f in input_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"\n⚠️  No images found in {input_dir}")
        print(f"   Supported formats: {', '.join(image_extensions)}")
        return
    
    print(f"\n{'=' * 80}")
    print("Image Transcription Pipeline")
    print(f"{'=' * 80}")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Images found:     {len(image_files)}")
    print(f"{'=' * 80}\n")
    
    # Process each image
    for image_path in sorted(image_files):
        try:
            process_image(image_path, output_dir)
        except Exception as e:
            print(f"  ✗ Error processing {image_path.name}: {e}\n")
            continue
    
    print(f"\n{'=' * 80}")
    print(f"✓ Processing complete! Transcriptions saved to: {output_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()

