# search_index.py
from __future__ import annotations
from typing import Optional, Callable, Dict, List
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import spacy

class SearchIndex:
    """
    Incremental hybrid (FAISS + BM25) index for notes.

    Usage:
        idx = SearchIndex()
        idx.build_from_db(db)                  # bootstrap once
        idx.add({"id": 1, "title": "...", "text": "..."})
        idx.edit({"id": 1, "title": "...", "text": "..."})
        idx.delete(1)
        results = idx.search("query", top_k=7, resolver=db.get)
    """

    SELF_PRONOUNS = {"i", "me", "my", "mine", "myself"}
    CONTENT_POS = {"NOUN", "PROPN", "ADJ"}

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        sbert_model: str = "paraphrase-MiniLM-L6-v2",
    ):
        # NLP pieces
        self._nlp = spacy.load(spacy_model)
        self._model = SentenceTransformer(sbert_model)

        # Index state
        self._faiss_index = None                    # faiss.IndexIDMap2 over IndexFlatIP
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_by_id: Dict[int, List[str]] = {}

        # embedding dimension
        self._dim = self._model.get_sentence_embedding_dimension()

    # -------- tokenization --------
    def _tokenize(self, s: Optional[str]) -> List[str]:
        s = s or ""
        doc = self._nlp(s.lower())
        toks = []
        for t in doc:
            if not t.is_alpha:
                continue
            if t.text in self.SELF_PRONOUNS:
                toks.append(t.text)
            elif t.pos_ in self.CONTENT_POS and not t.is_stop:
                toks.append(t.lemma_)
        return toks

    # -------- bootstrap from DB --------
    def build_from_db(self, db) -> None:
        """Build FAISS (with ids) + BM25 from current DB. Safe when DB is empty."""
        base = faiss.IndexFlatIP(self._dim)
        self._faiss_index = faiss.IndexIDMap2(base)
        self._tokenized_by_id.clear()

        data = db.as_dicts()  # [{"id","title","text"}, ...]
        if not data:
            # Empty corpus: keep FAISS empty and give BM25 a dummy doc to avoid ZeroDivisionError
            self._bm25 = BM25Okapi([["__empty__"]])
            return

        dense_texts = [f"{n['title']}. {n['text'] or ''}" for n in data]
        vecs = self._model.encode(dense_texts, convert_to_numpy=True, normalize_embeddings=True)
        ids = np.array([n["id"] for n in data], dtype=np.int64)
        self._faiss_index.add_with_ids(vecs, ids)

        self._tokenized_by_id = {
            n["id"]: self._tokenize(f"{n['title']} {n['text'] or ''}")
            for n in data
        }
        self._rebuild_bm25()

    # -------- incremental updates --------
    def add(self, note: Dict) -> None:
        """Add one note to both indices. note: {'id','title','text'}"""
        dense_text = f"{note['title']}. {note.get('text') or ''}"
        vec = self._model.encode([dense_text], convert_to_numpy=True, normalize_embeddings=True)
        self._faiss_index.add_with_ids(vec, np.array([note["id"]], dtype=np.int64))

        self._tokenized_by_id[note["id"]] = self._tokenize(f"{note['title']} {note.get('text') or ''}")
        self._rebuild_bm25()

    def delete(self, note_id: int) -> None:
        """Remove one note by id from both indices."""
        self._faiss_index.remove_ids(np.array([note_id], dtype=np.int64))
        self._tokenized_by_id.pop(note_id, None)
        self._rebuild_bm25()

    def edit(self, note: Dict) -> None:
        """Update one note: remove + re-add."""
        nid = note["id"]
        self._faiss_index.remove_ids(np.array([nid], dtype=np.int64))

        dense_text = f"{note['title']}. {note.get('text') or ''}"
        vec = self._model.encode([dense_text], convert_to_numpy=True, normalize_embeddings=True)
        self._faiss_index.add_with_ids(vec, np.array([nid], dtype=np.int64))

        self._tokenized_by_id[nid] = self._tokenize(f"{note['title']} {note.get('text') or ''}")
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        """Keep BM25 valid even when there are zero docs."""
        tokens_list = list(self._tokenized_by_id.values())
        if not tokens_list:
            # dummy doc prevents ZeroDivisionError inside rank-bm25
            self._bm25 = BM25Okapi([["__empty__"]])
        else:
            self._bm25 = BM25Okapi(tokens_list)

    # -------- search --------
    def search(
        self,
        query: str,
        top_k: int = 5,
        w_dense: float = 0.70,
        w_lex: float = 0.30,
        resolver: Optional[Callable[[int], object]] = None,
    ) -> List[Dict]:
        """
        Return [{'id', 'title', 'text', 'score'}] if resolver provided,
        else [{'id','score'}].
        resolver: callable that maps note_id -> Note (with .id .title .text)
        """
        if self._faiss_index is None or len(self._tokenized_by_id) == 0:
            return []

        id_list = list(self._tokenized_by_id.keys())   # defines the alignment for BM25
        if len(id_list) == 0:
            return []

        q_tokens = self._tokenize(query)

        # BM25 (aligned to id_list order)
        bm_scores = self._bm25.get_scores(q_tokens)
        bm_scores = bm_scores / (bm_scores.max() if bm_scores.max() > 0 else 1.0)

        # FAISS (returns ids because IndexIDMap2)
        N = len(id_list)
        qv = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self._faiss_index.search(qv, N)

        id_to_pos = {nid: i for i, nid in enumerate(id_list)}
        sem_scores = np.zeros(N, dtype=np.float32)
        for score, nid in zip(D[0], I[0]):
            pos = id_to_pos.get(int(nid))
            if pos is not None:
                sem_scores[pos] = score

        final = w_dense * sem_scores + w_lex * bm_scores
        order = np.argsort(final)[::-1][:top_k]

        results: List[Dict] = []
        for pos in order:
            nid = id_list[pos]
            if resolver:
                note = resolver(nid)
                if not note:
                    continue
                results.append({
                    "id": note.id,
                    "title": note.title,
                    "text": note.text or "",
                    "score": round(float(final[pos]), 3),   # 3 decimal places
                })
            else:
                results.append({"id": int(nid), "score": round(float(final[pos]), 3)})
        return results
