from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
from dotenv import load_dotenv


class EmbeddingError(RuntimeError):
    """Embedding provider failed or returned invalid data."""


class EmbeddingProvider(Protocol):
    name: str
    model: str

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class BaseHTTPEmbeddingProvider:
    """Shared HTTP behavior for embedding API providers."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        api_key: str,
        endpoint: str,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError(f"Missing API key for provider: {name}")

        self.name = name
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout

    def _request(
        self,
        texts: list[str],
        *,
        task: str | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        if any(not isinstance(text, str) or not text.strip()
               for text in texts):
            raise ValueError("Embedding inputs must be non-empty strings.")

        payload: dict = {
            "model": self.model,
            "input": texts,
        }

        if self.name == "jina":
            payload.update({
                "task": task,
                "normalized": True,
                "embedding_type": "float",
            })

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )

                response.raise_for_status()
                body = response.json()

        except httpx.HTTPStatusError as exc:
            # Do not include the request headers or API key in the error.
            raise EmbeddingError(
                f"{self.name} returned HTTP {exc.response.status_code}"
            ) from None

        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingError(
                f"Request to {self.name} failed: {type(exc).__name__}"
            ) from None

        try:
            rows = body["data"]

            # Some APIs return items in an order described by "index".
            rows = sorted(rows, key=lambda row: row.get("index", 0))

            vectors = [
                [float(value) for value in row["embedding"]]
                for row in rows
            ]

        except (KeyError, TypeError, ValueError):
            raise EmbeddingError(
                f"{self.name} returned an invalid embedding response."
            ) from None

        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"{self.name} returned {len(vectors)} vectors "
                f"for {len(texts)} inputs."
            )

        dimensions = {len(vector) for vector in vectors}

        if len(dimensions) != 1 or 0 in dimensions:
            raise EmbeddingError(
                f"{self.name} returned inconsistent vector dimensions."
            )

        if not np.isfinite(np.asarray(vectors)).all():
            raise EmbeddingError(
                f"{self.name} returned non-finite vector values."
            )

        return vectors


class JinaEmbeddingProvider(BaseHTTPEmbeddingProvider):
    def __init__(self) -> None:
        load_dotenv()

        super().__init__(
            name="jina",
            model=os.getenv(
                "JINA_EMBEDDING_MODEL",
                "jina-embeddings-v3",
            ),
            api_key=os.getenv("JINA_API_KEY", ""),
            endpoint="https://api.jina.ai/v1/embeddings",
        )

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return self._request(texts, task="retrieval.passage")

    def embed_query(self, text: str) -> list[float]:
        return self._request(
            [text],
            task="retrieval.query",
        )[0]


class OpenRouterEmbeddingProvider(BaseHTTPEmbeddingProvider):
    def __init__(self) -> None:
        load_dotenv()

        super().__init__(
            name="openrouter",
            model=os.getenv(
                "OPENROUTER_EMBEDDING_MODEL",
                "openai/text-embedding-3-small",
            ),
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            endpoint="https://openrouter.ai/api/v1/embeddings",
        )

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return self._request(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._request([text])[0]


class LocalEmbeddingProvider:
    """Optional CPU-capable Sentence Transformers implementation."""

    def __init__(self) -> None:
        load_dotenv()

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "Install local embedding support with: "
                "pip install sentence-transformers"
            ) from exc

        self.name = "local"
        self.model = os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            "intfloat/multilingual-e5-small",
        )
        self._model = SentenceTransformer(
            self.model,
            device="cpu",
        )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.astype(float).tolist()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if self.model.startswith("intfloat/multilingual-e5"):
            texts = [f"passage: {text}" for text in texts]

        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        if self.model.startswith("intfloat/multilingual-e5"):
            text = f"query: {text}"

        return self._encode([text])[0]


def create_embedding_provider() -> EmbeddingProvider:
    load_dotenv()

    provider = os.getenv(
        "RESEARCHPILOT_EMBEDDING_PROVIDER",
        "jina",
    ).lower()

    if provider == "jina":
        return JinaEmbeddingProvider()

    if provider == "openrouter":
        return OpenRouterEmbeddingProvider()

    if provider == "local":
        return LocalEmbeddingProvider()

    raise ValueError(
        "Unsupported embedding provider. "
        "Choose jina, openrouter, or local."
    )


class CachedEmbeddingProvider:
    """
    Persistent SQLite cache.

    The cache key includes provider, model, task and text.
    Query and document vectors are stored separately because
    some models use asymmetric retrieval embeddings.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache_path: str | Path | None = None,
    ) -> None:
        self.provider = provider

        self.name = provider.name
        self.model = provider.model

        load_dotenv()

        self.cache_path = Path(
            cache_path
            or os.getenv(
                "RESEARCHPILOT_EMBEDDING_CACHE",
                "data/embedding_cache.sqlite3",
            )
        )
        self.cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    role TEXT NOT NULL,
                    vector_json TEXT NOT NULL
                )
                """
            )

    def _key(self, role: str, text: str) -> str:
        payload = "\0".join([
            self.name,
            self.model,
            role,
            text,
        ])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _embed(
        self,
        texts: list[str],
        role: str,
    ) -> list[list[float]]:
        if not texts:
            return []

        keys = [self._key(role, text) for text in texts]

        with sqlite3.connect(self.cache_path) as conn:
            cached_rows = conn.execute(
                """
                SELECT cache_key, vector_json
                FROM embeddings
                WHERE cache_key IN (
                    """ + ",".join("?" for _ in set(keys)) + """
                )
                """,
                list(set(keys)),
            ).fetchall()

        cached = {
            key: json.loads(vector_json)
            for key, vector_json in cached_rows
        }

        # Preserve input order while embedding only cache misses.
        missing_text_by_key: dict[str, str] = {}

        for key, text in zip(keys, texts):
            if key not in cached:
                missing_text_by_key.setdefault(key, text)

        missing_keys = list(missing_text_by_key)
        missing_texts = [
            missing_text_by_key[key]
            for key in missing_keys
        ]

        if missing_texts:
            if role == "query":
                vectors = [
                    self.provider.embed_query(text)
                    for text in missing_texts
                ]
            else:
                # Bound request size; do not send the whole corpus
                # in one potentially enormous request.
                vectors = []

                for start in range(0, len(missing_texts), 32):
                    batch = missing_texts[start:start + 32]
                    vectors.extend(
                        self.provider.embed_documents(batch)
                    )

            if len(vectors) != len(missing_texts):
                raise EmbeddingError(
                    "Provider returned an unexpected vector count."
                )

            with sqlite3.connect(self.cache_path) as conn:
                for key, vector in zip(missing_keys, vectors):
                    vector_json = json.dumps(
                        vector,
                        separators=(",", ":"),
                    )

                    conn.execute(
                        """
                        INSERT OR REPLACE INTO embeddings
                        (cache_key, provider, model, role, vector_json)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            key,
                            self.name,
                            self.model,
                            role,
                            vector_json,
                        ),
                    )

                    cached[key] = vector

        result = [cached[key] for key in keys]

        dimensions = {len(vector) for vector in result}

        if len(dimensions) != 1:
            raise EmbeddingError(
                "Cached embeddings have inconsistent dimensions."
            )

        return result

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return self._embed(texts, role="passage")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], role="query")[0]