# pipeline/extractor.py
import pymupdf4llm
import uuid
import os

def extract_paper(filepath: str, filename: str) -> dict:
    """
    Stage 1: Extract text from PDF using pymupdf4llm.
    Returns paper dict with front, back, combined, readable flag.
    """
    try:
        markdown_text = pymupdf4llm.to_markdown(filepath)
    except Exception as e:
        # If extraction fails completely, treat as unreadable
        return {
            "paper_id": str(uuid.uuid4()),
            "filename": filename,
            "readable": False,
            "text": "",
            "front": "",
            "back": "",
            "combined": ""
        }

    total_length = len(markdown_text)
    if total_length < 500:
        # flag as scanned / unreadable
        return {
            "paper_id": str(uuid.uuid4()),
            "filename": filename,
            "readable": False,
            "text": markdown_text,  # keep for reference
            "front": "",
            "back": "",
            "combined": ""
        }

    front = markdown_text[:3000]
    back = markdown_text[-2000:] if total_length >= 2000 else markdown_text
    combined = front + "\n\n" + back

    return {
        "paper_id": str(uuid.uuid4()),
        "filename": filename,
        "readable": True,
        "text": markdown_text,
        "front": front,
        "back": back,
        "combined": combined
    }
