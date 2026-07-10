"""
Local, offline embedding backend used when no GOOGLE_API_KEY is configured.
Implements LangChain's Embeddings interface using TF-IDF vectors, so the
pipeline runs with zero API cost and zero internet dependency.
"""
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from langchain_core.embeddings import Embeddings


class LocalTfidfEmbeddings(Embeddings):
    def __init__(self, max_dimensions: int = 256, svd_threshold: int = 50):
        self.max_dimensions = max_dimensions
        self.svd_threshold = svd_threshold
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.svd = None
        self._fitted = False

    def _fit(self, texts: List[str]):
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        if tfidf_matrix.shape[0] >= self.svd_threshold:
            n_comp = min(self.max_dimensions, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
            self.svd = TruncatedSVD(n_components=max(n_comp, 1), random_state=42)
            self.svd.fit(tfidf_matrix)
        self._fitted = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            self._fit(texts)
        tfidf_matrix = self.vectorizer.transform(texts)
        vectors = self.svd.transform(tfidf_matrix) if self.svd else tfidf_matrix.toarray()
        return [self._normalize(v) for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        if not self._fitted:
            raise ValueError("Embedding model must be fit on documents before embedding queries.")
        tfidf_vec = self.vectorizer.transform([text])
        vector = self.svd.transform(tfidf_vec)[0] if self.svd else tfidf_vec.toarray()[0]
        return self._normalize(vector)

    @staticmethod
    def _normalize(vector) -> List[float]:
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()