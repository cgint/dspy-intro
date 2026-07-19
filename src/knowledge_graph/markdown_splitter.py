from typing import List, Optional
import re
import pydantic


class TextChunk(pydantic.BaseModel):
    """Represents a chunk of text extracted from markdown with metadata."""
    content: str = pydantic.Field(description="The actual text content of the chunk")
    chunk_type: str = pydantic.Field(description="Type of chunk: 'header_section', 'numbered_item', 'bulleted_item', 'paragraph'")
    header_context: Optional[str] = pydantic.Field(default=None, description="Parent header if applicable")
    chunk_index: int = pydantic.Field(description="Order of chunk in document (0-based)")


def _header_context_str(header_stack: List[str]) -> Optional[str]:
    """Build header context string from stack parents, e.g. 'Section > Subsection'.
    Uses header_stack[:-1] to exclude the current (deepest) header.
    Returns None when there are fewer than 2 headers (no parent context).
    """
    if len(header_stack) < 2:
        return None
    parts = [h.replace('#', '').strip() for h in header_stack[:-1]]
    return ' > '.join(parts)


def _collect_continuation(lines: List[str], start: int, stop_patterns: List[re.Pattern]) -> tuple[str, int]:
    """Scan from lines[start] onward, merging continuation lines into one text block.
    
    Stops at: empty line, any stop_pattern match, or end of lines.
    If a line starts with whitespace it's treated as continuation (stripped).
    Otherwise it's appended unmodified (heuristic for non-indented continuation).
    Returns (combined_text, new_index_after_consuming_lines).
    """
    content_parts: List[str] = []
    i = start
    while i < len(lines):
        next_line = lines[i].rstrip()
        if not next_line.strip():
            break
        if any(p.match(next_line) for p in stop_patterns):
            break
        if next_line.startswith(' ') or next_line.startswith('\t'):
            content_parts.append(next_line.strip())
        else:
            content_parts.append(next_line)
        i += 1
    return ' '.join(content_parts), i


def _emit_chunk(chunks: List[TextChunk], content: str, chunk_type: str, header_context: Optional[str], current_index: int) -> int:
    """Append a chunk if content >= 50 chars; return (possibly incremented) index."""
    if len(content.strip()) >= 50:
        chunks.append(TextChunk(
            content=content,
            chunk_type=chunk_type,
            header_context=header_context,
            chunk_index=current_index
        ))
        return current_index + 1
    return current_index


def split_markdown_into_chunks(text: str, strategy: str = "headers_first") -> List[TextChunk]:
    # lizard forgives(cyclomatic_complexity)  # essential: 3 content types + header nesting
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
                if header_pattern.match(next_line):
                    break
                if next_line.strip():
                    section_content.append(next_line)
                i += 1
            
            # Create chunk for header section if it has content
            if section_content:
                content = '\n'.join(section_content)
                hctx = _header_context_str(header_stack)
                
                # Further split by lists if present
                sub_chunks = _split_section_by_lists(content, hctx, current_chunk_index)
                if sub_chunks:
                    chunks.extend(sub_chunks)
                    current_chunk_index += len(sub_chunks)
                else:
                    current_chunk_index = _emit_chunk(chunks, content, "header_section", hctx, current_chunk_index)
            continue
        
        # Check for numbered list items (outside of header sections)
        numbered_match = numbered_item_pattern.match(line)
        if numbered_match:
            item_content = numbered_match.group(1).strip()
            continuation, i = _collect_continuation(lines, i + 1, [numbered_item_pattern, bulleted_item_pattern, header_pattern])
            if continuation:
                item_content += ' ' + continuation
            hctx = _header_context_str(header_stack)
            current_chunk_index = _emit_chunk(chunks, item_content, "numbered_item", hctx, current_chunk_index)
            continue
        
        # Check for bulleted list items
        bulleted_match = bulleted_item_pattern.match(line)
        if bulleted_match:
            item_content = bulleted_match.group(1).strip()
            continuation, i = _collect_continuation(lines, i + 1, [numbered_item_pattern, bulleted_item_pattern, header_pattern])
            if continuation:
                item_content += ' ' + continuation
            hctx = _header_context_str(header_stack)
            current_chunk_index = _emit_chunk(chunks, item_content, "bulleted_item", hctx, current_chunk_index)
            continue
        
        # Regular paragraph - collect until empty line or next special element
        paragraph_lines: List[str] = [line]
        continuation, i = _collect_continuation(lines, i + 1, [header_pattern, numbered_item_pattern, bulleted_item_pattern])
        if continuation:
            paragraph_lines.append(continuation)
        
        paragraph_content = ' '.join(paragraph_lines).strip()
        hctx = _header_context_str(header_stack)
        
        if len(paragraph_content.strip()) >= 50:
            if len(paragraph_content) > 2000:
                sentence_chunks = _split_large_text_by_sentences(paragraph_content, hctx, current_chunk_index)
                chunks.extend(sentence_chunks)
                current_chunk_index += len(sentence_chunks)
            else:
                current_chunk_index = _emit_chunk(chunks, paragraph_content, "paragraph", hctx, current_chunk_index)
    
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
            continuation, i = _collect_continuation(lines, i + 1, [numbered_pattern, bulleted_pattern])
            if continuation:
                item_content += ' ' + continuation
            chunk_index = _emit_chunk(chunks, item_content, "numbered_item", header_context, chunk_index)
        elif bulleted_match:
            item_content = bulleted_match.group(1).strip()
            continuation, i = _collect_continuation(lines, i + 1, [numbered_pattern, bulleted_pattern])
            if continuation:
                item_content += ' ' + continuation
            chunk_index = _emit_chunk(chunks, item_content, "bulleted_item", header_context, chunk_index)
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

