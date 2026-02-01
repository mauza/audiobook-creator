"""File format handlers for different input types."""

import re
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import PyPDF2
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from num2words import num2words

# Type alias for chapters: (title, text)
Chapter = Tuple[str, str]


def _normalize_for_tts(text: str) -> str:
    """Normalize text for natural TTS output.

    Converts currency, percentages, ordinals, numbers, and symbols
    to their spoken equivalents. Removes URLs and emails.

    Args:
        text: Raw text to normalize

    Returns:
        Normalized text suitable for TTS
    """
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # Currency: $42.50 -> forty-two dollars and fifty cents
    def _replace_currency(m):
        amount = m.group(0)[1:].replace(',', '')
        try:
            val = float(amount)
            dollars = int(val)
            cents = round((val - dollars) * 100)
            parts = []
            if dollars:
                parts.append(f"{num2words(dollars)} dollar{'s' if dollars != 1 else ''}")
            if cents:
                parts.append(f"{num2words(cents)} cent{'s' if cents != 1 else ''}")
            return ' and '.join(parts) if parts else 'zero dollars'
        except (ValueError, OverflowError):
            return ''

    text = re.sub(r'\$[\d,.]+', _replace_currency, text)

    # Percentages: 85% -> eighty-five percent
    def _replace_percent(m):
        try:
            return num2words(int(m.group(1))) + ' percent'
        except (ValueError, OverflowError):
            return m.group(0)

    text = re.sub(r'(\d+)%', _replace_percent, text)

    # Ordinals: 1st, 2nd, 3rd, 4th etc.
    def _replace_ordinal(m):
        try:
            return num2words(int(m.group(1)), to='ordinal')
        except (ValueError, OverflowError):
            return m.group(0)

    text = re.sub(r'(\d+)(?:st|nd|rd|th)\b', _replace_ordinal, text)

    # Plain numbers: 1,200 -> one thousand two hundred
    def _replace_number(m):
        try:
            num_str = m.group(0).replace(',', '')
            return num2words(int(num_str))
        except (ValueError, OverflowError):
            return m.group(0)

    text = re.sub(r'\d[\d,]+', _replace_number, text)

    # Single remaining digits
    def _replace_single_digit(m):
        try:
            return num2words(int(m.group(0)))
        except (ValueError, OverflowError):
            return m.group(0)

    text = re.sub(r'\b\d+\b', _replace_single_digit, text)

    # Symbols
    text = text.replace(' & ', ' and ')
    text = text.replace(' + ', ' plus ')
    text = text.replace(' = ', ' equals ')
    text = re.sub(r'(?<!\w)@(?!\w)', 'at', text)

    # Strip leftover non-speakable chars (keep letters, digits, basic punctuation, whitespace)
    text = re.sub(r"[^\w\s.,;:!?'\"-]", '', text)

    # Collapse multiple spaces
    text = re.sub(r'  +', ' ', text)

    return text.strip()


def _strip_markdown(text: str) -> str:
    """Strip markdown formatting from text.

    Args:
        text: Markdown-formatted text

    Returns:
        Plain text with markdown formatting removed
    """
    # Remove images ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Convert links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove code fences
    text = re.sub(r'```[^\n]*\n(.*?)```', r'\1', text, flags=re.DOTALL)
    # Remove inline backticks
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove bold **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    # Remove italic *text* or _text_ (but not underscores inside words)
    text = re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'\1', text)
    text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'\1', text)
    # Strip heading markers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r'^[-*]{3,}\s*$', '', text, flags=re.MULTILINE)

    return text


def _detect_page_headers_footers(pages: List[str]) -> Set[str]:
    """Detect repeated headers and footers across PDF pages.

    Args:
        pages: List of text content per page

    Returns:
        Set of normalized line patterns that appear on >50% of pages
    """
    if len(pages) < 4:
        return set()

    candidate_counts: dict[str, int] = {}

    for page_text in pages:
        lines = page_text.strip().split('\n')
        # Take first 2 and last 2 lines as candidates
        candidates = []
        candidates.extend(lines[:2])
        if len(lines) > 2:
            candidates.extend(lines[-2:])

        seen_on_page: set[str] = set()
        for line in candidates:
            normalized = re.sub(r'\b\d+\b', 'N', line.strip())
            if normalized and normalized not in seen_on_page:
                seen_on_page.add(normalized)
                candidate_counts[normalized] = candidate_counts.get(normalized, 0) + 1

    threshold = len(pages) * 0.5
    return {pattern for pattern, count in candidate_counts.items() if count > threshold}


def _clean_text(text: str) -> str:
    """Clean extracted text by handling common issues.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    # Fix common hyphenation at line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # Normalize multiple newlines to double newline (preserve paragraph breaks)
    text = re.sub(r'\n{2,}', '\n\n', text)
    # Normalize spaces within each line (but preserve newlines)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while handling common abbreviations.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    # First, temporarily replace abbreviations with a special marker
    abbreviations = {
        'Mr.': 'MR_ABBR',
        'Mrs.': 'MRS_ABBR',
        'Dr.': 'DR_ABBR',
        'Prof.': 'PROF_ABBR',
        'St.': 'ST_ABBR',
        'Ave.': 'AVE_ABBR',
        'Blvd.': 'BLVD_ABBR',
        'Rd.': 'RD_ABBR',
        'Inc.': 'INC_ABBR',
        'Ltd.': 'LTD_ABBR',
        'Co.': 'CO_ABBR',
        'Corp.': 'CORP_ABBR',
        'vs.': 'VS_ABBR',
        'e.g.': 'EG_ABBR',
        'i.e.': 'IE_ABBR',
        'etc.': 'ETC_ABBR',
        'approx.': 'APPROX_ABBR',
        'dept.': 'DEPT_ABBR',
        'est.': 'EST_ABBR',
        'min.': 'MIN_ABBR',
        'max.': 'MAX_ABBR',
        'no.': 'NO_ABBR',
        'tel.': 'TEL_ABBR',
        'temp.': 'TEMP_ABBR',
        'vol.': 'VOL_ABBR',
        'fig.': 'FIG_ABBR',
        'ref.': 'REF_ABBR',
        'p.': 'P_ABBR',
        'pp.': 'PP_ABBR',
    }

    # Replace abbreviations with markers
    for abbr, marker in abbreviations.items():
        text = text.replace(abbr, marker)

    # Split on sentence endings
    sentences = re.split(r'[.!?]\s+', text)

    # Restore abbreviations
    for abbr, marker in abbreviations.items():
        for i, sentence in enumerate(sentences):
            sentences[i] = sentence.replace(marker, abbr)

    # Add back the punctuation
    matches = re.finditer(r'[.!?]\s+', text)
    punctuation = [m.group().strip() for m in matches]

    # Combine sentences with their punctuation
    result = []
    for i, sentence in enumerate(sentences):
        if i < len(punctuation):
            result.append(sentence + punctuation[i])
        else:
            result.append(sentence)

    return [s.strip() for s in result if s.strip()]


def extract_text_from_pdf(
    file_path: Path,
    chapter_aware: bool = False,
) -> Union[str, List[Chapter]]:
    """Extract text from a PDF file.

    Args:
        file_path: Path to the PDF file
        chapter_aware: If True, attempt to split into chapters using PDF outline

    Returns:
        Extracted text, or list of (title, text) chapters if chapter_aware
    """
    pages_text: List[str] = []
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)

        for page in pdf_reader.pages:
            pages_text.append(page.extract_text() or '')

        # Remove repeated headers/footers
        repeated = _detect_page_headers_footers(pages_text)
        if repeated:
            cleaned_pages = []
            for page_text in pages_text:
                lines = page_text.split('\n')
                filtered = []
                for line in lines:
                    normalized = re.sub(r'\b\d+\b', 'N', line.strip())
                    if normalized not in repeated:
                        filtered.append(line)
                cleaned_pages.append('\n'.join(filtered))
            pages_text = cleaned_pages

        if chapter_aware and pdf_reader.outline:
            # Try to map outline entries to page numbers
            chapters: List[Chapter] = []

            def _flatten_outline(outline, result=None):
                if result is None:
                    result = []
                for item in outline:
                    if isinstance(item, list):
                        _flatten_outline(item, result)
                    else:
                        try:
                            page_num = pdf_reader.get_destination_page_number(item)
                            result.append((item.title, page_num))
                        except Exception:
                            pass
                return result

            flat_outline = _flatten_outline(pdf_reader.outline)

            if flat_outline:
                for i, (title, start_page) in enumerate(flat_outline):
                    end_page = flat_outline[i + 1][1] if i + 1 < len(flat_outline) else len(pages_text)
                    chapter_text = ' '.join(pages_text[start_page:end_page])
                    chapter_text = _clean_text(chapter_text)
                    sentences = _split_into_sentences(chapter_text)
                    chapters.append((title, ' '.join(sentences)))

                if chapters:
                    return chapters

        # Flat text fallback
        full_text = ' '.join(pages_text)
        full_text = _clean_text(full_text)
        sentences = _split_into_sentences(full_text)
        flat = ' '.join(sentences)

        if chapter_aware:
            return [("Full Text", flat)]
        return flat


def extract_text_from_epub(
    file_path: Path,
    chapter_aware: bool = False,
    skip_front_matter: bool = True,
) -> Union[str, List[Chapter]]:
    """Extract text from an EPUB file.

    Args:
        file_path: Path to the EPUB file
        chapter_aware: If True, split into chapters based on headings
        skip_front_matter: If True, skip cover, TOC, copyright, etc.

    Returns:
        Extracted text, or list of (title, text) chapters if chapter_aware
    """
    FRONT_MATTER_PATTERNS = [
        'cover', 'toc', 'nav', 'copyright', 'titlepage',
        'frontmatter', 'dedication', 'halftitle',
    ]

    book = epub.read_epub(file_path)

    # Get spine-ordered items
    spine_ids = [item_id for item_id, _ in book.spine]
    items_by_id = {item.get_id(): item for item in book.get_items()}
    spine_items = [items_by_id[sid] for sid in spine_ids if sid in items_by_id]

    # Filter front matter
    if skip_front_matter:
        filtered = []
        for item in spine_items:
            item_id = (item.get_id() or '').lower()
            item_href = (item.get_name() or '').lower()
            identifier = item_id + ' ' + item_href
            if not any(pat in identifier for pat in FRONT_MATTER_PATTERNS):
                filtered.append(item)
        spine_items = filtered

    if chapter_aware:
        chapters: List[Chapter] = []
        current_title = None
        current_text_parts: List[str] = []

        for item in spine_items:
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()

            # Look for chapter headings
            headings = soup.find_all(['h1', 'h2'])
            if headings:
                for heading in headings:
                    # Save previous chapter
                    if current_title is not None and current_text_parts:
                        ch_text = _clean_text(' '.join(current_text_parts))
                        sentences = _split_into_sentences(ch_text)
                        chapters.append((current_title, ' '.join(sentences)))

                    current_title = heading.get_text(strip=True)
                    # Get text after this heading within the same document
                    current_text_parts = []
                    for sibling in heading.find_next_siblings():
                        current_text_parts.append(sibling.get_text(separator=' '))
            else:
                # No heading, append to current chapter
                current_text_parts.append(soup.get_text(separator=' '))

        # Save last chapter
        if current_text_parts:
            title = current_title or "Untitled"
            ch_text = _clean_text(' '.join(current_text_parts))
            sentences = _split_into_sentences(ch_text)
            chapters.append((title, ' '.join(sentences)))

        if chapters:
            return chapters
        # Fallback: no headings found

    # Flat text extraction
    text = []
    for item in spine_items:
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text.append(soup.get_text(separator=' '))

    full_text = ' '.join(text)
    full_text = _clean_text(full_text)
    sentences = _split_into_sentences(full_text)
    flat = ' '.join(sentences)

    if chapter_aware:
        return [("Full Text", flat)]
    return flat


def extract_text_from_file(
    file_path: Path,
    encoding: str = 'utf-8',
    chapter_aware: bool = False,
    skip_front_matter: bool = True,
) -> Union[str, List[Chapter]]:
    """Extract text from a file based on its extension.

    Args:
        file_path: Path to the input file
        encoding: File encoding for text files
        chapter_aware: If True, attempt to split into chapters
        skip_front_matter: If True, skip front matter in EPUB files

    Returns:
        Extracted text, or list of (title, text) chapters if chapter_aware

    Raises:
        ValueError: If the file format is not supported
    """
    suffix = file_path.suffix.lower()

    if suffix == '.pdf':
        return extract_text_from_pdf(file_path, chapter_aware=chapter_aware)
    elif suffix == '.epub':
        return extract_text_from_epub(
            file_path,
            chapter_aware=chapter_aware,
            skip_front_matter=skip_front_matter,
        )
    elif suffix == '.md':
        text = file_path.read_text(encoding=encoding)
        text = _strip_markdown(text)
        text = _clean_text(text)

        if chapter_aware:
            # Split on top-level headings (already stripped of #)
            # Re-read raw to find heading boundaries
            raw = file_path.read_text(encoding=encoding)
            parts = re.split(r'^#\s+(.+)$', raw, flags=re.MULTILINE)
            if len(parts) > 1:
                chapters: List[Chapter] = []
                # parts[0] is text before first heading (may be empty)
                i = 1
                while i < len(parts) - 1:
                    title = parts[i].strip()
                    body = _strip_markdown(parts[i + 1])
                    body = _clean_text(body)
                    sentences = _split_into_sentences(body)
                    chapters.append((title, ' '.join(sentences)))
                    i += 2
                if chapters:
                    return chapters

            sentences = _split_into_sentences(text)
            return [("Full Text", ' '.join(sentences))]

        sentences = _split_into_sentences(text)
        return ' '.join(sentences)
    elif suffix == '.txt':
        text = file_path.read_text(encoding=encoding)
        text = _clean_text(text)
        sentences = _split_into_sentences(text)
        flat = ' '.join(sentences)
        if chapter_aware:
            return [("Full Text", flat)]
        return flat
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
