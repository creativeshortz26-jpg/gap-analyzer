# pipeline/chunker.py
def chunk_text(text: str, chunk_size_words=200, overlap_words=50, min_words=80) -> list:
    """
    Stage 2: Split combined text into word-based overlapping chunks.
    Discard chunks smaller than min_words.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size_words
        chunk_words = words[start:end]
        if len(chunk_words) < min_words:
            break
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        start += (chunk_size_words - overlap_words)
        if start >= len(words):
            break
    return chunks

def chunk_papers(papers: list) -> list:
    """
    For each readable paper, chunk its combined text.
    Returns list of chunk dicts.
    """
    all_chunks = []
    for paper in papers:
        if not paper.get("readable", False):
            continue
        text = paper["combined"]
        raw_chunks = chunk_text(text)
        for i, chunk_txt in enumerate(raw_chunks):
            all_chunks.append({
                "chunk_id": f"{paper['paper_id']}_chunk_{i}",
                "paper_id": paper["paper_id"],
                "filename": paper["filename"],
                "text": chunk_txt
            })
    return all_chunks
