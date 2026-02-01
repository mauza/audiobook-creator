"""File format handlers for different input types."""

import re
from pathlib import Path
from typing import Optional, List
import PyPDF2
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup


def _clean_text(text: str) -> str:
    """Clean extracted text by handling common issues.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Replace multiple newlines with double newline
    text = re.sub(r'\n+', '\n\n', text)
    # Fix common hyphenation at line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # Fix common hyphenation at page breaks
    text = re.sub(r'(\w+)-\s*(\w+)', r'\1\2', text)
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


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
    text = []
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text.append(page.extract_text())
    
    # Join all pages and clean the text
    full_text = ' '.join(text)
    full_text = _clean_text(full_text)
    
    # Split into sentences and rejoin with proper spacing
    sentences = _split_into_sentences(full_text)
    return ' '.join(sentences)


def extract_text_from_epub(file_path: Path) -> str:
    """Extract text from an EPUB file.
    
    Args:
        file_path: Path to the EPUB file
        
    Returns:
        Extracted text from the EPUB
    """
    book = epub.read_epub(file_path)
    text = []
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            # Get text
            text.append(soup.get_text(separator=' '))
    
    # Join all sections and clean the text
    full_text = ' '.join(text)
    full_text = _clean_text(full_text)
    
    # Split into sentences and rejoin with proper spacing
    sentences = _split_into_sentences(full_text)
    return ' '.join(sentences)


def extract_text_from_file(file_path: Path, encoding: str = 'utf-8') -> str:
    """Extract text from a file based on its extension.
    
    Args:
        file_path: Path to the input file
        encoding: File encoding for text files
        
    Returns:
        Extracted text from the file
        
    Raises:
        ValueError: If the file format is not supported
    """
    suffix = file_path.suffix.lower()
    
    if suffix == '.pdf':
        return extract_text_from_pdf(file_path)
    elif suffix == '.epub':
        return extract_text_from_epub(file_path)
    elif suffix in ['.txt', '.md']:
        text = file_path.read_text(encoding=encoding)
        text = _clean_text(text)
        sentences = _split_into_sentences(text)
        return ' '.join(sentences)
    else:
        raise ValueError(f"Unsupported file format: {suffix}") 