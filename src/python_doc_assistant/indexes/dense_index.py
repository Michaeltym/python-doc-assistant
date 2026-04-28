"""Dense embedding index for v2 retrieval (plan v2 §1).

Wraps `sentence-transformers` (`BAAI/bge-small-en-v1.5` by default) +
numpy. Build once from list[Chunk]; `search()` returns top-K DenseHits
ranked by cosine similarity.

Persistence layout (mirrors `BM25Index` sha-keyed convention):
    data/indexes/<DOCS_VERSION>/<sha_short>/dense.npy   — N x D float32
    data/indexes/<DOCS_VERSION>/<sha_short>/dense.json  — sidecar
        {"model_id": str, "chunk_ids": list[str], "dim": int}

NOTE: `sentence_transformers` and `numpy` are *lazily* imported inside
the methods that need them — importing this module without the
`embedding` extra installed must NOT raise. (Tests rely on this.)
"""

from __future__ import annotations

import json  # noqa: F401  (used by save/load implementations — see docstring)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from python_doc_assistant.ingest.chunker import Chunk

# ------------------------------------------------------------------
# Defaults (plan v2 §1)
# ------------------------------------------------------------------

DEFAULT_MODEL_ID: Final[str] = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_DIM: Final[int] = 384  # bge-small-en-v1.5 hidden size


# ------------------------------------------------------------------
# Data records
# ------------------------------------------------------------------


@dataclass(frozen=True)
class DenseHit:
    """One dense-retrieval result."""

    chunk_id: str
    score: float  # cosine similarity in [-1, 1]; with normalized embeddings


# ------------------------------------------------------------------
# Public: dense index
# ------------------------------------------------------------------


class DenseIndex:
    """Cosine-similarity index over L2-normalized chunk embeddings.

    Build path:
        index = DenseIndex(chunks)
        index.save(Path("data/indexes/3.12/<sha>/dense.npy"))

    Load + query path:
        index = DenseIndex.load(Path("data/indexes/3.12/<sha>/dense.npy"))
        hits  = index.search("how to read a file", k=5)

    Test path (DI):
        index = DenseIndex(chunks, model=stub_encoder)   # no model load
    """

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        model_id: str = DEFAULT_MODEL_ID,
        model: Any = None,
    ) -> None:
        """Encode each chunk's `title + text`, store as N x D float32 array.

        Args:
            chunks: list of Chunk objects to encode.
            model_id: HuggingFace model id (used when `model` is None).
            model: pre-loaded SentenceTransformer (or stub for tests). When
                provided, `from_pretrained` is skipped entirely.

        Implementation outline:
            1. self._model_id = model_id
            2. self._chunk_ids: list[str] = [c.chunk_id for c in chunks]
            3. if model is None:
                   from sentence_transformers import SentenceTransformer
                   model = SentenceTransformer(model_id)
               self._model = model
            4. texts = [self._chunk_to_text(c) for c in chunks]
            5. self._embeddings = self._model.encode(
                   texts,
                   normalize_embeddings=True,
                   convert_to_numpy=True,
                   show_progress_bar=False,
               )
               # shape: (N, D)
        """
        self._model_id = model_id
        self._chunk_ids = [c.chunk_id for c in chunks]
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_id)
            self._model = model
        else:
            self._model = model
        texts = [self._chunk_to_text(c) for c in chunks]
        if not texts:
            import numpy as np

            self._embeddings = np.empty((0, DEFAULT_EMBEDDING_DIM), dtype=np.float32)
        else:
            self._embeddings = self._model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

    def search(self, query: str, *, k: int = 10) -> list[DenseHit]:
        """Top-K DenseHits ranked by cosine similarity (highest first).

        With L2-normalized embeddings, cosine == inner product, so:

            scores  = self._embeddings @ query_emb        # shape: (N,)
            top_idx = np.argsort(-scores)[:k]
            return [DenseHit(self._chunk_ids[i], float(scores[i])) for i in top_idx]

        Notes:
            - Encode the query with the same `normalize_embeddings=True`
              flag so the inner product equals cosine.
            - Empty corpus → return [].
            - No score-floor filter; cosine is unbounded below; downstream
              hybrid merge can apply its own threshold.
        """
        if len(self._chunk_ids) == 0:
            return []
        import numpy as np

        query_embedding = self._model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        scores = self._embeddings @ query_embedding
        top_idx = np.argsort(-scores)[:k]
        return [DenseHit(chunk_id=self._chunk_ids[i], score=float(scores[i])) for i in top_idx]

    def save(self, path: Path) -> None:
        """Persist embeddings to `<path>` and sidecar metadata to `<path>.json`.

        Implementation outline:
            1. path.parent.mkdir(parents=True, exist_ok=True)
            2. np.save(path, self._embeddings)
            3. meta_path = path.with_suffix(".json")
            4. meta_path.write_text(json.dumps({
                   "model_id": self._model_id,
                   "chunk_ids": self._chunk_ids,
                   "dim": int(self._embeddings.shape[1]),
               }))

        `path` is conventionally `dense.npy`; the sidecar becomes `dense.json`.
        """
        import numpy as np

        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self._embeddings)
        meta_path = path.with_suffix(".json")
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_id": self._model_id,
                    "chunk_ids": self._chunk_ids,
                    "dim": int(self._embeddings.shape[1]),
                },
                f,
            )

    @classmethod
    def load(cls, path: Path, *, model: Any = None) -> DenseIndex:
        """Restore from `<path>` (.npy) + `<path>.json` (sidecar).

        Args:
            path: path to the .npy file.
            model: pre-loaded SentenceTransformer (test path). When None,
                lazy-load via `sentence_transformers.SentenceTransformer(meta["model_id"])`.

        Raises:
            FileNotFoundError if either `path` or its `.json` sidecar is missing.

        Implementation outline:
            1. validate both files exist
            2. embeddings = np.load(path)
            3. meta = json.loads(meta_path.read_text())
            4. instance = cls.__new__(cls)
               instance._model_id   = meta["model_id"]
               instance._chunk_ids  = meta["chunk_ids"]
               instance._embeddings = embeddings
               instance._model      = model or SentenceTransformer(meta["model_id"])
               return instance
        """

        if not path.exists():
            raise FileNotFoundError(f"Embeddings not found at {path}")
        json_path = path.with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata not found at {json_path}")
        import numpy as np

        embeddings = np.load(path)
        with json_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        instance = cls.__new__(cls)
        instance._embeddings = embeddings
        instance._model_id = metadata["model_id"]
        instance._chunk_ids = metadata["chunk_ids"]
        if model is not None:
            instance._model = model
        else:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(instance._model_id)
            instance._model = model
        return instance

    # ------------------------------------------------------------------
    # Helpers (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_to_text(chunk: Chunk) -> str:
        """Compose the encoder input from a chunk.

        Default: `<title>\\n\\n<text>`. Title disambiguates chunks with
        similar bodies (e.g. multiple `read_text` methods across modules).
        """
        return f"{chunk.title}\n\n{chunk.text}"
