import argparse
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Callable, Generator, assert_never

import git
from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.ollama.document_embedder import (
    OllamaDocumentEmbedder,
)

from talk_rag_scheduler.argument_parser import add_indexing_args, parse_indexing_args
from talk_rag_scheduler.const import (
    INDEXING_MODE,
    OLLAMA_EMBED_URL,
    OLLAMA_EMBEDDING_MODEL,
    OUT_PATH,
    SCHEDULER_SRC_PATH,
    SPLIT_BY,
    SPLIT_LENGTH,
    SPLIT_OVERLAP,
    GIT_REMOTE,
    SPLIT_THRESHOLD,
)
from talk_rag_scheduler.models import T_INDEXING_MODE, T_SPLIT_BY

KEYWORD_INDEXING_PIPE_IMG = OUT_PATH / "keyword_indexing_pipeline.png"
SEMANTIC_INDEXING_PIPE_IMG = OUT_PATH / "semantic_indexing_pipeline.png"
SEMANTIC_SPLIT_INDEXING_PIPE_IMG = OUT_PATH / "semantic_split_indexing_pipeline.png"


def is_file_empty(file_path: Path) -> bool:
    path = Path(file_path)
    return path.stat().st_size == 0


def get_files(
    directory: Path,
    exclude_dirs: Callable[[Path], bool] | None = None,
    include_files: Callable[[Path], bool] | None = None,
) -> Generator[Path, None, None]:
    for item in directory.iterdir():
        if (
            item.is_file()
            and not is_file_empty(item)
            and (include_files is None or include_files(item))
        ):
            yield item
        elif item.is_dir() and (exclude_dirs is None or not exclude_dirs(item)):
            yield from get_files(
                item, exclude_dirs=exclude_dirs, include_files=include_files
            )


def remote_files_generator() -> Generator[Path, None, None]:
    return get_files(
        SCHEDULER_SRC_PATH,
        exclude_dirs=lambda path: path.name == ".git",
        include_files=lambda path: path.suffix
        in [".py", ".md", ".rst", ".toml", ".txt", ".yaml", "LICENSE"],
    )


def create_keyword_indexing_pipeline(
    document_store: InMemoryDocumentStore,
) -> Pipeline:
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("converter", TextFileToDocument())
    indexing_pipeline.add_component(
        "writer", DocumentWriter(document_store=document_store)
    )
    indexing_pipeline.connect("converter", "writer")
    return indexing_pipeline


def create_vector_indexing_pipeline(
    document_store: InMemoryDocumentStore,
    ollama_embedding_model: str = OLLAMA_EMBEDDING_MODEL,
) -> Pipeline:
    vector_indexing_pipeline = Pipeline()
    vector_indexing_pipeline.add_component("converter", TextFileToDocument())
    vector_indexing_pipeline.add_component(
        "embedder",
        OllamaDocumentEmbedder(model=ollama_embedding_model, url=OLLAMA_EMBED_URL),
    )
    vector_indexing_pipeline.add_component(
        "writer", DocumentWriter(document_store=document_store)
    )
    vector_indexing_pipeline.connect("converter", "embedder")
    vector_indexing_pipeline.connect("embedder", "writer")
    return vector_indexing_pipeline


def create_splitting_vector_indexing_pipeline(
    document_store: InMemoryDocumentStore,
    ollama_embedding_model: str = OLLAMA_EMBEDDING_MODEL,
    split_by: T_SPLIT_BY = SPLIT_BY,
    split_length: int = SPLIT_LENGTH,
    split_overlap: int = SPLIT_OVERLAP,
    split_threshold: int = SPLIT_THRESHOLD,
) -> Pipeline:
    splitting_vector_indexing_pipeline = Pipeline()
    splitting_vector_indexing_pipeline.add_component("converter", TextFileToDocument())
    splitting_vector_indexing_pipeline.add_component(
        "splitter",
        DocumentSplitter(
            split_by=split_by,
            split_length=split_length,
            split_overlap=split_overlap,
            split_threshold=split_threshold,
        ),
    )
    splitting_vector_indexing_pipeline.add_component(
        "embedder",
        OllamaDocumentEmbedder(model=ollama_embedding_model, url=OLLAMA_EMBED_URL),
    )
    splitting_vector_indexing_pipeline.add_component(
        "writer", DocumentWriter(document_store=document_store)
    )
    splitting_vector_indexing_pipeline.connect("converter", "splitter")
    splitting_vector_indexing_pipeline.connect("splitter", "embedder")
    splitting_vector_indexing_pipeline.connect("embedder", "writer")
    return splitting_vector_indexing_pipeline


def ingest_knowledge_and_run_indexing(
    document_store: InMemoryDocumentStore,
    indexing_mode: T_INDEXING_MODE = INDEXING_MODE,
    ollama_embedding_model: str = OLLAMA_EMBEDDING_MODEL,
    split_by: T_SPLIT_BY = SPLIT_BY,
    split_length: int = SPLIT_LENGTH,
    split_overlap: int = SPLIT_OVERLAP,
    split_threshold: int = SPLIT_THRESHOLD,
) -> Pipeline:
    if not SCHEDULER_SRC_PATH.exists():
        git.Repo.clone_from(GIT_REMOTE, SCHEDULER_SRC_PATH)
    source_files = list(remote_files_generator())
    OUT_PATH.mkdir(exist_ok=True)
    match indexing_mode:
        case "keyword":
            indexing_pipeline = create_keyword_indexing_pipeline(
                document_store=document_store,
            )
            indexing_pipeline.draw(KEYWORD_INDEXING_PIPE_IMG)
        case "semantic":
            indexing_pipeline = create_vector_indexing_pipeline(
                document_store=document_store,
                ollama_embedding_model=ollama_embedding_model,
            )
            indexing_pipeline.draw(SEMANTIC_INDEXING_PIPE_IMG)
        case "semantic_split":
            indexing_pipeline = create_splitting_vector_indexing_pipeline(
                document_store=document_store,
                ollama_embedding_model=ollama_embedding_model,
                split_by=split_by,
                split_length=split_length,
                split_overlap=split_overlap,
                split_threshold=split_threshold,
            )
            indexing_pipeline.draw(SEMANTIC_SPLIT_INDEXING_PIPE_IMG)
        case _ as unreachable:
            assert_never(unreachable)
            raise ValueError(f"Unknown mode {indexing_mode}")

    # TODO: consider splitting out
    indexing_pipeline.run(
        {
            "converter": {
                "sources": source_files,
                "meta": [{"file.suffix": file.suffix} for file in source_files],
            }
        }
    )
    return indexing_pipeline


def main() -> None:
    parser = ArgumentParser(
        prog="Indexing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Index the scheduler documents",
        epilog="""Example usage:
  python -m talk_rag_scheduler.indexing_pipeline --indexing_mode semantic
  python -m talk_rag_scheduler.indexing_pipeline --indexing_mode semantic_split --split_length 12
""",
    )
    add_indexing_args(parser)
    indexing_kwargs = parse_indexing_args(parser.parse_args())
    document_store = InMemoryDocumentStore()
    indexing_pipeline = ingest_knowledge_and_run_indexing(
        document_store, **indexing_kwargs
    )


if __name__ == "__main__":
    main()
