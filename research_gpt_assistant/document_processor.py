# document_processor.py
import os
import re
import logging
from math import ceil

import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DocumentProcessor:
    """
    DocumentProcessor - simple, beginner-friendly implementation.
    Stores processed documents in self.documents:
      { doc_id: {'title': str, 'chunks': [str,...], 'metadata': {...}} }
    """

    def __init__(self, config):
        self.config = config
        # fallback defaults if not set in config
        self.CHUNK_SIZE = getattr(config, "CHUNK_SIZE", 1000)
        self.OVERLAP = getattr(config, "OVERLAP", 100)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                          max_features=5000,
                                          stop_words="english")
        self.documents = {}
        # flat lists built when build_search_index() is called
        self.chunk_texts = []
        self.chunk_doc_ids = []
        self.document_vectors = None
        self.logger = getattr(config, "logger", logging.getLogger(__name__))

    # PDF extraction & cleaning
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from PDF file and lightly clean it.
        Returns cleaned string (maybe empty string if extraction failed).
        """
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
        """
        Basic cleaning:
          - fix hyphenation at line breaks: "exam-\nple" -> "example"
          - remove control characters
          - turn newlines into spaces and collapse repeated whitespace
        """
        if not text:
            return ""

        # remove common control characters and replace windows newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # remove hyphenation caused by line breaks
        text = re.sub(r"-\n\s*", "", text)

        # replace newlines with spaces
        text = text.replace("\n", " ")

        # remove non-printable/control characters
        text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)

        # collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # chunking
    def chunk_text(self, text, chunk_size=None, overlap=None):
        """
        Create overlapping chunks from text. Chunk sizes are in characters.
        Strategy:
          - Use a sliding-window approach (step = chunk_size - overlap)
          - When possible, extend chunk end to the next sentence boundary
        """
        if not text:
            return []

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

            # try to expand 'end' to end of sentence (look ahead up to 300 chars)
            if end < n:
                lookahead = text[end : min(n, end + 300)]
                m = re.search(r"[.!?][\"']?\s", lookahead)
                if m:
                    end += m.end()

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= n:
                break
            # next window starts 'step' chars after start
            start += step

        return chunks

    # Processing pipeline
    def process_document(self, pdf_path):
        """
        Full pipeline for one PDF:
          - extract text
          - preprocess
          - chunk
          - extract basic metadata (title, num_pages, words, chars)
          - store under a unique doc_id
        Returns the doc_id.
        """
        # canonical doc_id (filename without extension). If collision, append suffix.
        base = os.path.basename(pdf_path)
        doc_id_base = os.path.splitext(base)[0]
        doc_id = doc_id_base
        i = 1
        while doc_id in self.documents:
            doc_id = f"{doc_id_base}_{i}"
            i += 1

        # read pages and metadata (also used for title)
        try:
            reader = PyPDF2.PdfReader(pdf_path)
            num_pages = len(reader.pages)
            # metadata could be None or a dictionary-like object
            meta = getattr(reader, "metadata", {})
            title = None
            try:
                # try common metadata forms
                if meta:
                    title = getattr(meta, "title", None) or meta.get("/Title") or meta.get("Title")
            except Exception:
                title = None
        except Exception:
            # fall back if reader fails
            num_pages = 0
            title = None

        cleaned_text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(cleaned_text)

        # fallback title to filename if metadata title missing
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

        self.logger.info("Processed '%s' -> %d chunks, %d words", doc_id, len(chunks), metadata["num_words"])
        return doc_id

    # search index
    def build_search_index(self):
        """
        Build TF-IDF index from all chunks of stored documents.
        After calling this:
          - self.chunk_texts: list of all chunks
          - self.chunk_doc_ids: corresponding doc IDs
          - self.document_vectors: sparse matrix (TF-IDF)
        """
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

        # fit/transform
        self.vectorizer.fit(all_chunks)
        vectors = self.vectorizer.transform(all_chunks)

        self.chunk_texts = all_chunks
        self.chunk_doc_ids = chunk_doc_ids
        self.document_vectors = vectors
        self.logger.info("Built TF-IDF index: %d chunks indexed", len(all_chunks))

    def find_similar_chunks(self, query, top_k=5):
        """
        Return list of (chunk_text, similarity_score, doc_id) sorted by score desc.
        """
        if self.document_vectors is None or not self.chunk_texts:
            raise RuntimeError("Search index is not built. Call build_search_index() first.")

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

    # stats
    def get_document_stats(self):
        """
        Return simple stats: number of documents, total chunks, average doc length (words), titles.
        """
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
