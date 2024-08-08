from typing import Literal, TypedDict

from haystack.dataclasses.document import Document


class MetaType(TypedDict):
    model: str
    created_at: str
    done: bool
    done_reason: str
    context: list[int]
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int


class LLMOutput(TypedDict):
    replies: list[str]
    meta: list[MetaType]


class RagOutput(TypedDict):
    llm: LLMOutput


class ConcurrentRagOutput(TypedDict):
    llms: dict[str, LLMOutput]
    t_total: float


class DocumentResults(TypedDict):
    documents: list[Document]


class RetrievalResult(TypedDict):
    retriever: DocumentResults


class ExtendedConfusionMatrix(TypedDict):
    TP: int
    FP: int
    TN: int
    FN: int
    ACC: float
    F1: float
    TPR: float  # Recall
    # PPV - not that helpful, as we return fixed number of results no matter what
    PPV: float  # Precision


class GoldenRetrieval(TypedDict):
    query: str
    golden_file_paths: list[str]


T_INDEXING_MODE = Literal["keyword", "semantic", "semantic_split"]
T_SPLIT_BY = Literal["word", "sentence", "page", "passage"]


class IndexingKwargs(TypedDict):
    indexing_mode: T_INDEXING_MODE
    split_by: T_SPLIT_BY
    split_length: int
    split_overlap: int
    split_threshold: int
    ollama_embedding_model: str


class _RetrievalKwargs(TypedDict):
    query: str
    top_k: int


class RetrievalKwargs(TypedDict):
    indexing_mode: T_INDEXING_MODE
    query: str
    top_k: int


class RagPipelineKwargs(_RetrievalKwargs, IndexingKwargs):
    ollama_model: str
    num_predict: int
    temperature: float


class ConcurrentRagPipelineKwargs(RagPipelineKwargs):
    n_concurrent: int
