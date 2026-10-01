"""
MILESTONE 2 - Document Processing
---------------------------------
Load PDF / TXT / DOCX -> clean text -> create RAG chunks.
"""
import os
import re
from pypdf import PdfReader

def load_document(path):
    """Load text from PDF, TXT, or DOCX."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(path)
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )
    if ext == ".txt":
        with open(path, encoding="utf-8") as f:
            return f.read()
    if ext == ".docx":
        from docx import Document
        return "\n\n".join(
            p.text for p in Document(path).paragraphs
            if p.text.strip()
        )
    raise ValueError(f"Unsupported file type: {ext}")

def clean_text(text):
    """Remove common PDF extraction noise while preserving paragraphs."""
    # Join words split across PDF lines: "safe-\nty" -> "safety"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Remove page-number-only lines
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    # Normalize spaces/tabs without destroying newlines
    text = re.sub(r"[ \t]+", " ", text)
    # Reduce excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def make_chunks(text, size=500, overlap=50):
    """
    Split text by paragraphs first.
    Long paragraphs are divided into overlapping chunks.
    """
    if size <= overlap:
        raise ValueError("size must be greater than overlap")

    chunks = []
    paragraphs = re.split(r"\n{2,}", text)
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= size + 300:
            chunks.append(para)
            continue
        start = 0
        while start < len(para):
            chunk = para[start:start + size].strip()
            if chunk:
                chunks.append(chunk)
            start += size - overlap

    return chunks

def process_file(path):
    """Complete document-processing pipeline."""
    text = load_document(path)
    text = clean_text(text)
    return make_chunks(text)

if __name__ == "__main__":
    path = "data/sample_safety_manual.txt"
    chunks = process_file(path)
    print(f"{len(chunks)} chunks created")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i} ---")
        print(chunk[:500])