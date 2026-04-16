"""Text chunker for documents using sliding window with overlap."""

import re
from typing import List, Dict

class DocumentChunker:
    """Chunks text into manageable pieces for embedding."""

    def __init__(self, max_tokens: int = 512, overlap: int = 64):
        # A rough heuristic: 1 token ≈ 4 characters
        self.chunk_size = max_tokens * 4
        self.chunk_overlap = overlap * 4

    def chunk_page(self, page_dict: Dict) -> List[Dict]:
        """
        Chunks text from a single page preserving page layout/origin.
        Expects page_dict: {"page_num": int, "text": str, ...}
        """
        text = page_dict.get("text", "")
        if not text:
            return []

        # Split into paragraphs to try and respect boundaries
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk_text = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds size, save current chunk and start new
            if len(current_chunk_text) + len(para) > self.chunk_size and current_chunk_text:
                chunks.append({
                    "page_number": page_dict.get("page_num"),
                    "text": current_chunk_text.strip(),
                    "ocr_confidence": page_dict.get("ocr_confidence")
                })
                # Overlap logic: keep last portion of current text
                overlap_text = current_chunk_text[-self.chunk_overlap:]
                # Try to snap overlap to a word boundary
                snap_idx = overlap_text.find(" ")
                if snap_idx != -1 and snap_idx < len(overlap_text) - 10:
                    overlap_text = overlap_text[snap_idx:].strip()
                
                current_chunk_text = overlap_text + "\n\n" + para
            else:
                if current_chunk_text:
                    current_chunk_text += "\n\n" + para
                else:
                    current_chunk_text = para
                    
            # If a single paragraph is too huge, split it blindly
            while len(current_chunk_text) > self.chunk_size:
                chunks.append({
                    "page_number": page_dict.get("page_num"),
                    "text": current_chunk_text[:self.chunk_size].strip(),
                    "ocr_confidence": page_dict.get("ocr_confidence")
                })
                current_chunk_text = current_chunk_text[self.chunk_size - self.chunk_overlap:]

        if current_chunk_text:
            chunks.append({
                "page_number": page_dict.get("page_num"),
                "text": current_chunk_text.strip(),
                "ocr_confidence": page_dict.get("ocr_confidence")
            })

        return chunks

    def chunk_document(self, pages: List[Dict]) -> List[Dict]:
        """Chunks a list of pages and assigns global chunk indices."""
        all_chunks = []
        chunk_idx = 0
        
        for page in pages:
            page_chunks = self.chunk_page(page)
            for c in page_chunks:
                c["chunk_index"] = chunk_idx
                chunk_idx += 1
                all_chunks.append(c)
                
        return all_chunks
