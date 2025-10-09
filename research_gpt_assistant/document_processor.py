# document_processor.py
import os
import pandas as pd
import re
import logging
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config as config
import logging

class DocumentProcessor:
    def __init__(self, config):
      """
    DocumentProcessor - beginner-friendly implementation.
    Stores processed documents in self.documents:
      { doc_id: {'title': str, 'chunks': [str,...], 'metadata': {...}} }
    """
      self.config = config
        # fallback defaults if not set in config
      self.CHUNK_SIZE = getattr(config, "CHUNK_SIZE", 1000)
      self.OVERLAP = getattr(config, "OVERLAP", 100)
      self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            stop_words="english"
        )
      self.documents = {}
      self.chunk_texts = []
      self.chunk_doc_ids = []
      self.document_vectors = None
      self.logger = getattr(config, "logger", logging.getLogger(__name__))

    # PDF Extraction and Preprocessing
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file and clean it."""
        try:
            reader = PyPDF2.PdfReader(pdf_path)
        except Exception as e:
            self.logger.exception("Failed to open PDF %s: %s", pdf_path, e)
            return ""

        raw_pages = []
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
            except Exception:
                page_text = None
            if page_text:
                raw_pages.append(page_text)

        raw_text = "\n".join(raw_pages)
        cleaned = self.preprocess_text(raw_text)
        return cleaned

    def preprocess_text(self, text):
        """Clean up PDF text."""
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"-\n\s*", "", text)
        text = text.replace("\n", " ")
        text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Chunking
    def chunk_text(self, text, chunk_size=None, overlap=None):
        """Split long text into overlapping chunks."""
        if chunk_size is None:
            chunk_size = self.CHUNK_SIZE
        if overlap is None:
            overlap = self.OVERLAP

        text = text.strip()
        n = len(text)
        if n <= chunk_size:
            return [text]

        step = max(chunk_size - overlap, 1)
        chunks = []
        start = 0
        while start < n:
            end = min(start + chunk_size, n)

            # Try to end at sentence boundary
            if end < n:
                lookahead = text[end:min(n, end + 300)]
                m = re.search(r"[.!?][\"']?\s", lookahead)
                if m:
                    end += m.end()

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= n:
                break
            start += step

        return chunks

    def get_document_chunks(self, doc_id):
        """Return list of text chunks for a specific document."""
        if doc_id not in self.documents:
            return []
        return self.documents[doc_id].get("chunks", [])

    # Processing Pipeline
    def process_document(self, pdf_path):
        """Extract, clean, chunk, and store a document."""
        base = os.path.basename(pdf_path)
        doc_id_base = os.path.splitext(base)[0]
        doc_id = doc_id_base
        i = 1
        while doc_id in self.documents:
            doc_id = f"{doc_id_base}_{i}"
            i += 1

        try:
            reader = PyPDF2.PdfReader(pdf_path)
            num_pages = len(reader.pages)
            meta = getattr(reader, "metadata", {})
            title = None
            if meta:
                try:
                    title = getattr(meta, "title", None) or meta.get("/Title") or meta.get("Title")
                except Exception:
                    title = None
        except Exception:
            num_pages = 0
            title = None

        cleaned_text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(cleaned_text)

        if not title:
            title = doc_id_base

        metadata = {
            "path": pdf_path,
            "num_pages": num_pages,
            "num_words": len(cleaned_text.split()),
            "num_chars": len(cleaned_text),
        }

        self.documents[doc_id] = {
            "title": title,
            "chunks": chunks,
            "metadata": metadata,
        }

        self.logger.info(
            "Processed '%s' -> %d chunks, %d words",
            doc_id, len(chunks), metadata["num_words"]
        )
        return doc_id

    # Search Index
    def build_search_index(self):
        """Build TF-IDF index for all document chunks."""
        all_chunks = []
        chunk_doc_ids = []
        for doc_id, doc in self.documents.items():
            for chunk in doc["chunks"]:
                all_chunks.append(chunk)
                chunk_doc_ids.append(doc_id)

        if not all_chunks:
            self.logger.warning("No chunks to index. Run process_document() first.")
            self.chunk_texts = []
            self.chunk_doc_ids = []
            self.document_vectors = None
            return

        self.vectorizer.fit(all_chunks)
        vectors = self.vectorizer.transform(all_chunks)

        self.chunk_texts = all_chunks
        self.chunk_doc_ids = chunk_doc_ids
        self.document_vectors = vectors
        self.logger.info("Built TF-IDF index: %d chunks indexed", len(all_chunks))

    def find_similar_chunks(self, query, top_k=5):
        """Return list of (chunk_text, similarity_score, doc_id)."""
        if self.document_vectors is None or not self.chunk_texts:
            raise RuntimeError("Search index not built. Call build_search_index() first.")

        cleaned_q = self.preprocess_text(query)
        q_vec = self.vectorizer.transform([cleaned_q])
        scores = cosine_similarity(q_vec, self.document_vectors)[0]
        if len(scores) == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            results.append((self.chunk_texts[idx], score, self.chunk_doc_ids[idx]))
        return results

   
    # Statistics
    def get_document_stats(self):
        """Return simple statistics about stored documents."""
        num_docs = len(self.documents)
        total_chunks = sum(len(d["chunks"]) for d in self.documents.values())
        avg_len = 0
        titles = []
        if num_docs > 0:
            total_words = sum(d["metadata"].get("num_words", 0) for d in self.documents.values())
            avg_len = total_words / num_docs
            titles = [d["title"] for d in self.documents.values()]

        return {
            "num_documents": num_docs,
            "total_chunks": total_chunks,
            "avg_document_length_words": avg_len,
            "titles": titles,
        }
