from argparse import ArgumentParser, Namespace
from typing import assert_never

from talk_rag_scheduler.const import (
    INDEXING_MODE,
    N_CONCURRENT,
    NUM_PREDICT,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_MODEL,
    QUERY_0,
    SPLIT_BY,
    SPLIT_LENGTH,
    SPLIT_OVERLAP,
    SPLIT_THRESHOLD,
    TEMPERATURE,
    TOP_K,
)
from talk_rag_scheduler.models import (
    ConcurrentRagPipelineKwargs,
    IndexingKwargs,
    RagPipelineKwargs,
    RetrievalKwargs,
)


def add_indexing_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--indexing_mode",
        choices=["keyword", "semantic", "semantic_split"],
        default=None,
        help=f"Defaults to {INDEXING_MODE} if not specified",
    )
    parser.add_argument(
        "--split_by",
        choices=["word", "sentence", "page", "passage"],
        default=None,
        help=f"Splitting unit used for semantic_split, defaults to {SPLIT_BY}",
    )
    parser.add_argument(
        "--split_length",
        type=int,
        default=None,
        help=f"Splitting length used for semantic_split, defaults to {SPLIT_LENGTH}",
    )
    parser.add_argument(
        "--split_overlap",
        type=int,
        default=None,
        help=f"Splitting overlap used for semantic_split, defaults to {SPLIT_OVERLAP}",
    )
    parser.add_argument(
        "--split_threshold",
        type=int,
        default=None,
        help=f"Splitting threshold used for semantic_split, defaults to {SPLIT_THRESHOLD}",
    )
    parser.add_argument("--ollama_embedding_model", type=str, default=None)


def add_retrieval_args(parser: ArgumentParser) -> None:
    add_indexing_args(parser)
    parser.add_argument(
        "query", nargs="?", type=str, help=f"The question to ask, defaluts to {QUERY_0}"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help=f"Number of documents to retrieve, defaults to {TOP_K}",
    )


def add_pipeline_args(parser: ArgumentParser) -> None:
    add_retrieval_args(parser)
    parser.add_argument(
        "--ollama_model",
        type=str,
        default=None,
        help=f"Ollama large language model to use, defaults to {OLLAMA_MODEL}",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--num_predict", type=int, default=None)


def add_concurrent_pipeline_args(parser: ArgumentParser) -> None:
    add_pipeline_args(parser)
    parser.add_argument("--n_concurrent", type=int, default=None)


def parse_indexing_args(namespace: Namespace) -> IndexingKwargs:
    indexing_kwargs: IndexingKwargs
    match namespace:
        case Namespace(
            indexing_mode="keyword" | "semantic" | None as indexing_mode,
            split_by=None,
            split_length=None,
            split_overlap=None,
            split_threshold=None,
            ollama_embedding_model=None,
        ):
            indexing_kwargs = {
                "indexing_mode": indexing_mode or INDEXING_MODE,
                "split_by": SPLIT_BY,
                "split_length": SPLIT_LENGTH,
                "split_overlap": SPLIT_OVERLAP,
                "split_threshold": SPLIT_THRESHOLD,
                "ollama_embedding_model": OLLAMA_EMBEDDING_MODEL,
            }
        case Namespace(
            indexing_mode="keyword" | "semantic" | None,
        ):
            raise ValueError(
                "Conflict: split_by, split_length, split_overlap and split_threshold are only valid with option --indexing_mode semantic_split"
            )
        case Namespace(
            indexing_mode=indexing_mode,
            split_by=split_by,
            split_length=split_length,
            split_overlap=split_overlap,
            split_threshold=split_threshold,
            ollama_embedding_model=ollama_embedding_model,
        ):
            indexing_kwargs = {
                "indexing_mode": indexing_mode or INDEXING_MODE,
                "split_by": split_by or SPLIT_BY,
                "split_length": split_length or SPLIT_LENGTH,
                "split_overlap": split_overlap or SPLIT_OVERLAP,
                "split_threshold": split_threshold or SPLIT_THRESHOLD,
                "ollama_embedding_model": ollama_embedding_model
                or OLLAMA_EMBEDDING_MODEL,
            }
        case _ as unreachable:
            assert_never(unreachable)
            raise ValueError("Unexpected Arguments")

    return indexing_kwargs


def parse_retrieval_kwargs(namespace: Namespace) -> RetrievalKwargs:
    retrieval_kwargs: RetrievalKwargs
    match namespace:
        case Namespace(
            indexing_mode=indexing_mode,
            query=query,
            top_k=top_k,
        ):
            retrieval_kwargs = {
                "indexing_mode": indexing_mode or INDEXING_MODE,
                "query": query or QUERY_0,
                "top_k": top_k or TOP_K,
            }
        case _ as unreachable:
            assert_never(unreachable)
            raise ValueError("Invalid arguments")
    return retrieval_kwargs


def parse_pipeline_args(namespace: Namespace) -> RagPipelineKwargs:
    indexing_kwargs: IndexingKwargs = parse_indexing_args(namespace)
    retrieval_kwargs: RetrievalKwargs = parse_retrieval_kwargs(namespace)
    pipeline_kwargs: RagPipelineKwargs
    match namespace:
        case Namespace(
            ollama_model=ollama_model,
            temperature=temperature,
            num_predict=num_predict,
        ):
            pipeline_kwargs = {
                **indexing_kwargs,
                **retrieval_kwargs,
                "ollama_model": ollama_model or OLLAMA_MODEL,
                "temperature": temperature or TEMPERATURE,
                "num_predict": num_predict or NUM_PREDICT,
            }
        case _ as unreachable:
            assert_never(unreachable)
            raise ValueError("Invalid arguments")

    return pipeline_kwargs


def parse_concurrent_pipeline_args(namespace: Namespace) -> ConcurrentRagPipelineKwargs:
    pipeline_kwargs: RagPipelineKwargs = parse_pipeline_args(namespace)
    match namespace:
        case Namespace(n_concurrent=n_concurrent):
            return {**pipeline_kwargs, "n_concurrent": n_concurrent or N_CONCURRENT}
        case _ as unreachable:
            assert_never(unreachable)
            raise ValueError("Invalid arguments")
