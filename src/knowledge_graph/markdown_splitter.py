from typing import List, Optional
import re
import pydantic


class TextChunk(pydantic.BaseModel):
    """Represents a chunk of text extracted from markdown with metadata."""
    content: str = pydantic.Field(description="The actual text content of the chunk")
    chunk_type: str = pydantic.Field(description="Type of chunk: 'header_section', 'numbered_item', 'bulleted_item', 'paragraph'")
    header_context: Optional[str] = pydantic.Field(default=None, description="Parent header if applicable")
    chunk_index: int = pydantic.Field(description="Order of chunk in document (0-based)")


def split_markdown_into_chunks(text: str, strategy: str = "headers_first") -> List[TextChunk]:
    """
    Split markdown text into logical chunks for better triplet extraction.
    
    Strategy: headers_first
    - Priority order: Headers (H1-H6) → Numbered lists → Bulleted lists → Paragraphs
    - Each header creates a new section
    - Numbered/bulleted items are split individually
    - Remaining text is split into paragraphs
    
    Args:
        text: The markdown text to split
        strategy: Splitting strategy (currently only "headers_first" is supported)
    
    Returns:
        List of TextChunk objects with metadata
    """
    chunks: List[TextChunk] = []
    lines = text.split('\n')
    
    # Track current header context (stack for nested headers)
    header_stack: List[str] = []
    current_chunk_index = 0
    
    # Patterns for markdown elements
    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
    numbered_item_pattern = re.compile(r'^\d+\.\s+(.+)$')
    bulleted_item_pattern = re.compile(r'^[-*+]\s+(.+)$')
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Skip empty lines
        if not line.strip():
            i += 1
            continue
        
        # Check for headers (H1-H6)
        header_match = header_pattern.match(line)
        if header_match:
            header_level = len(header_match.group(1))
            
            # Update header stack - remove headers at same or deeper level
            header_stack = [h for h in header_stack if h.count('#') < header_level]
            header_stack.append(line)
            
            # Collect content until next header or end
            section_content: List[str] = []
            i += 1
            
            while i < len(lines):
                next_line = lines[i].rstrip()
                
                # Stop at next header
                if header_pattern.match(next_line):
                    break
                
                # Collect non-empty lines
                if next_line.strip():
                    section_content.append(next_line)
                
                i += 1
            
            # Create chunk for header section if it has content
            if section_content:
                content = '\n'.join(section_content)
                header_context = ' > '.join([h.replace('#', '').strip() for h in header_stack[:-1]]) if len(header_stack) > 1 else None
                
                # Further split by lists if present
                sub_chunks = _split_section_by_lists(content, header_context, current_chunk_index)
                if sub_chunks:
                    chunks.extend(sub_chunks)
                    current_chunk_index += len(sub_chunks)
                else:
                    # No lists found, treat as paragraph
                    if len(content.strip()) >= 50:  # Skip very small chunks
                        chunks.append(TextChunk(
                            content=content,
                            chunk_type="header_section",
                            header_context=header_context,
                            chunk_index=current_chunk_index
                        ))
                        current_chunk_index += 1
            continue
        
        # Check for numbered list items (outside of header sections)
        numbered_match = numbered_item_pattern.match(line)
        if numbered_match:
            item_content = numbered_match.group(1).strip()
            
            # Collect continuation lines (indented or part of same item)
            i += 1
            while i < len(lines):
                next_line = lines[i].rstrip()
                if not next_line.strip():
                    break
                # If next line is another list item or header, stop
                if numbered_item_pattern.match(next_line) or bulleted_item_pattern.match(next_line) or header_pattern.match(next_line):
                    break
                # If indented, it's continuation
                if next_line.startswith(' ') or next_line.startswith('\t'):
                    item_content += ' ' + next_line.strip()
                else:
                    # Might be continuation or new paragraph
                    item_content += ' ' + next_line
                i += 1
            
            header_context = ' > '.join([h.replace('#', '').strip() for h in header_stack]) if header_stack else None
            
            if len(item_content.strip()) >= 50:  # Skip very small chunks
                chunks.append(TextChunk(
                    content=item_content,
                    chunk_type="numbered_item",
                    header_context=header_context,
                    chunk_index=current_chunk_index
                ))
                current_chunk_index += 1
            continue
        
        # Check for bulleted list items
        bulleted_match = bulleted_item_pattern.match(line)
        if bulleted_match:
            item_content = bulleted_match.group(1).strip()
            
            # Collect continuation lines
            i += 1
            while i < len(lines):
                next_line = lines[i].rstrip()
                if not next_line.strip():
                    break
                if numbered_item_pattern.match(next_line) or bulleted_item_pattern.match(next_line) or header_pattern.match(next_line):
                    break
                if next_line.startswith(' ') or next_line.startswith('\t'):
                    item_content += ' ' + next_line.strip()
                else:
                    item_content += ' ' + next_line
                i += 1
            
            header_context = ' > '.join([h.replace('#', '').strip() for h in header_stack]) if header_stack else None
            
            if len(item_content.strip()) >= 50:  # Skip very small chunks
                chunks.append(TextChunk(
                    content=item_content,
                    chunk_type="bulleted_item",
                    header_context=header_context,
                    chunk_index=current_chunk_index
                ))
                current_chunk_index += 1
            continue
        
        # Regular paragraph - collect until empty line or next special element
        paragraph_lines: List[str] = [line]
        i += 1
        
        while i < len(lines):
            next_line = lines[i].rstrip()
            if not next_line.strip():
                break
            # Stop at headers or list items
            if header_pattern.match(next_line) or numbered_item_pattern.match(next_line) or bulleted_item_pattern.match(next_line):
                break
            paragraph_lines.append(next_line)
            i += 1
        
        paragraph_content = '\n'.join(paragraph_lines).strip()
        header_context = ' > '.join([h.replace('#', '').strip() for h in header_stack]) if header_stack else None
        
        if len(paragraph_content) >= 50:  # Skip very small chunks
            # Split very large paragraphs by sentences
            if len(paragraph_content) > 2000:
                sentence_chunks = _split_large_text_by_sentences(paragraph_content, header_context, current_chunk_index)
                chunks.extend(sentence_chunks)
                current_chunk_index += len(sentence_chunks)
            else:
                chunks.append(TextChunk(
                    content=paragraph_content,
                    chunk_type="paragraph",
                    header_context=header_context,
                    chunk_index=current_chunk_index
                ))
                current_chunk_index += 1
    
    return chunks


def _split_section_by_lists(content: str, header_context: Optional[str], start_index: int) -> List[TextChunk]:
    """Split a section content by numbered or bulleted lists."""
    chunks: List[TextChunk] = []
    lines = content.split('\n')
    
    numbered_pattern = re.compile(r'^\d+\.\s+(.+)$')
    bulleted_pattern = re.compile(r'^[-*+]\s+(.+)$')
    
    i = 0
    chunk_index = start_index
    
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        numbered_match = numbered_pattern.match(line)
        bulleted_match = bulleted_pattern.match(line)
        
        if numbered_match:
            item_content = numbered_match.group(1).strip()
            i += 1
            
            # Collect continuation
            while i < len(lines):
                next_line = lines[i].rstrip()
                if not next_line.strip():
                    break
                if numbered_pattern.match(next_line) or bulleted_pattern.match(next_line):
                    break
                if next_line.startswith(' ') or next_line.startswith('\t'):
                    item_content += ' ' + next_line.strip()
                else:
                    item_content += ' ' + next_line
                i += 1
            
            if len(item_content.strip()) >= 50:
                chunks.append(TextChunk(
                    content=item_content,
                    chunk_type="numbered_item",
                    header_context=header_context,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
        elif bulleted_match:
            item_content = bulleted_match.group(1).strip()
            i += 1
            
            while i < len(lines):
                next_line = lines[i].rstrip()
                if not next_line.strip():
                    break
                if numbered_pattern.match(next_line) or bulleted_pattern.match(next_line):
                    break
                if next_line.startswith(' ') or next_line.startswith('\t'):
                    item_content += ' ' + next_line.strip()
                else:
                    item_content += ' ' + next_line
                i += 1
            
            if len(item_content.strip()) >= 50:
                chunks.append(TextChunk(
                    content=item_content,
                    chunk_type="bulleted_item",
                    header_context=header_context,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
        else:
            i += 1
    
    return chunks


def _split_large_text_by_sentences(text: str, header_context: Optional[str], start_index: int) -> List[TextChunk]:
    """Split very large text chunks by sentences (approximately 2000 chars per chunk)."""
    chunks: List[TextChunk] = []
    
    # Simple sentence splitting (period followed by space or newline)
    sentences = re.split(r'([.!?]\s+)', text)
    
    # Recombine sentences with their punctuation
    combined_sentences: List[str] = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            combined_sentences.append(sentences[i] + sentences[i + 1])
        else:
            combined_sentences.append(sentences[i])
    
    if len(sentences) % 2 == 1:
        combined_sentences.append(sentences[-1])
    
    current_chunk = ""
    chunk_index = start_index
    
    for sentence in combined_sentences:
        if len(current_chunk) + len(sentence) > 2000 and current_chunk:
            chunks.append(TextChunk(
                content=current_chunk.strip(),
                chunk_type="paragraph",
                header_context=header_context,
                chunk_index=chunk_index
            ))
            chunk_index += 1
            current_chunk = sentence
        else:
            current_chunk += sentence
    
    if current_chunk.strip():
        chunks.append(TextChunk(
            content=current_chunk.strip(),
            chunk_type="paragraph",
            header_context=header_context,
            chunk_index=chunk_index
        ))
    
    return chunks

