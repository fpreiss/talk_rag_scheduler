from pathlib import Path
from typing import Final

from talk_rag_scheduler.models import T_INDEXING_MODE, T_SPLIT_BY, GoldenRetrieval

FILE_PARENT_PATH: Final[Path] = Path(__file__).parent.parent.parent.resolve()
DATA_SRC_PATH: Final[Path] = (FILE_PARENT_PATH / "data").resolve()
SCHEDULER_SRC_PATH: Final[Path] = DATA_SRC_PATH / "scheduler"
OUT_PATH: Final[Path] = FILE_PARENT_PATH / "out"
OLLAMA_HOST: Final[str] = "http://127.0.0.1:11434"
OLLAMA_OPENAI_URL: Final[str] = f"{OLLAMA_HOST}/v1"
OLLAMA_GENERATE_URL: Final[str] = f"{OLLAMA_HOST}/api/generate"
OLLAMA_EMBED_URL: Final[str] = f"{OLLAMA_HOST}/api/embeddings"
# OLLAMA_MODEL:Final[str] = "codestral:22b-v0.1-q8_0"
OLLAMA_MODEL: Final[str] = "gemma2:2b"
NUM_PREDICT: Final[int] = 500
TEMPERATURE: Final[float] = 0.3
TOP_K: Final[int] = 3
N_CONCURRENT: Final[int] = 5

SPLIT_LENGTH: Final[int] = 10
SPLIT_OVERLAP: Final[int] = 3
SPLIT_THRESHOLD: Final[int] = 6
SPLIT_BY: Final[T_SPLIT_BY] = "passage"
OLLAMA_EMBEDDING_MODEL: Final[str] = "mxbai-embed-large"
INDEXING_MODE: Final[T_INDEXING_MODE] = "keyword"
GIT_REMOTE: Final[str] = "https://gitlab.com/DigonIO/scheduler.git"

GOLDEN_RETRIEVALS: Final[list[GoldenRetrieval]] = [
    {
        "query": "How can I schedule an asynchronous job using the scheduler python library?",
        "golden_file_paths": [
            "scheduler/asyncio/scheduler.py",
            "scheduler/threading/scheduler.py",
            "doc/pages/examples/threading.rst",
            "doc/pages/examples/asyncio.rst",
            "tests/threading/scheduler/test_sch_threading.py",
            "tests/asyncio/test_async_scheduler.py",
            "tests/asyncio/test_async_scheduler_cyclic.py",
        ],
    }
]

QUERY_0: Final[str] = GOLDEN_RETRIEVALS[0]["query"]
PROMPT_TEMPLATE: Final[
    str
] = """
    Answer the query based on the provided context.
    If the context does not contain the answer, say 'Answer not found'.
    Context:
    {% for doc in documents %}
    {{ doc.content }}
    {% endfor %}
    query: {{query}}
    Answer:
    """
