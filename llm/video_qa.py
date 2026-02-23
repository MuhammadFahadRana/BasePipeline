"""
Video QA System

Combines semantic search and LLM to answer natural language questions about video content.
Functions as a RAG (Retrieval-Augmented Generation) system.
"""

import torch
import json
import time
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from pathlib import Path

from search.semantic_search import SemanticSearchEngine, SearchResult

class VideoQA:
    """
    RAG-based Question Answering for videos.
    Retrieves relevant snippets (transcript, OCR, visual semantics) and generates an answer.
    """
    
    DEFAULT_QA_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    
    def __init__(
        self, 
        db: Session, 
        model_name: str = DEFAULT_QA_MODEL,
        device: str = "auto"
    ):
        """
        Initialize Video QA system.
        
        Args:
            db: Database session
            model_name: LLM for answer generation
            device: "auto", "cuda", or "cpu"
        """
        self.db = db
        self.search_engine = SemanticSearchEngine(db)
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading QA LLM: {model_name}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")
            
        print(f"✓ Video QA system ready ({model_name})")

    def ask(
        self, 
        question: str, 
        video_filter: Optional[str] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Answer a question about the video content.
        
        Args:
            question: User's question
            video_filter: Optional specific video filename
            top_k: Number of context snippets to retrieve
            
        Returns:
            Dict with 'answer', 'citations', and 'metadata'
        """
        start_time = time.time()
        
        # 1. Retrieve relevant context using the enhanced search engine
        # This will find matches in Transcripts, OCR, and Visual Captions/Labels
        search_results = self.search_engine.search(
            query=question,
            top_k=top_k,
            video_filter=video_filter,
            semantic_weight=0.7,
            text_weight=0.3
        )
        
        if not search_results:
            return {
                "answer": "I couldn't find any relevant information in the video library to answer your question. Try rephrasing or asking about specific objects or topics.",
                "citations": [],
                "metadata": {"search_query": question, "elapsed_ms": (time.time() - start_time) * 1000}
            }
            
        # 2. Construct context from results
        context_parts = []
        for i, res in enumerate(search_results):
            # Format: [Source 1 at 00:12:34]: Text content
            timestamp = f"{int(res.start_time // 3600):02d}:{int((res.start_time % 3600) // 60):02d}:{int(res.start_time % 60):02d}"
            source_tag = f"[Source {i+1} at {timestamp} in {res.video_filename}]"
            context_parts.append(f"{source_tag}\n{res.text}")
            
        context_text = "\n\n".join(context_parts)
        
        # 3. Build prompt
        prompt = self._build_prompt(question, context_text)
        
        # 4. Generate answer
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1, # Keep it factual
                do_sample=False,
                repetition_penalty=1.1
            )
            
        # Decode only the generated part
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            "answer": response.strip(),
            "citations": [res.to_dict() for res in search_results],
            "metadata": {
                "search_query": question,
                "elapsed_ms": round(elapsed_ms, 2),
                "context_used": len(search_results)
            }
        }

    def _build_prompt(self, question: str, context: str) -> str:
        """Construct the RAG prompt."""
        return f"""You are a specialized Video Assistant. Your goal is to answer questions about video content based ONLY on the provided snippets (Transcripts, OCR text, and Visual descriptions).

RULES:
1. Use ONLY the provided context snippets to answer.
2. If the answer isn't in the snippets, say you don't know based on the available data.
3. Reference the sources (e.g., [Source 1]) when stating facts from them.
4. Be concise and professional.

CONTEXT SNIPPETS:
{context}

QUESTION: {question}

ANSWER:"""

if __name__ == "__main__":
    # Quick test if run as script
    from database.config import SessionLocal
    
    print("Testing Video QA System...")
    db = SessionLocal()
    qa = VideoQA(db)
    
    test_q = "What objects are visible in the oil rig scenes?"
    result = qa.ask(test_q)
    
    print(f"\nQ: {test_q}")
    print(f"A: {result['answer']}")
    print(f"\nCitations: {len(result['citations'])} sources used.")
    
    db.close()
