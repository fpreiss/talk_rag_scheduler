import argparse
from argparse import ArgumentParser

from haystack import Pipeline
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.ollama.text_embedder import (
    OllamaTextEmbedder,
)

from talk_rag_scheduler.argument_parser import (
    add_retrieval_args,
    parse_indexing_args,
    parse_retrieval_kwargs,
)
from talk_rag_scheduler.const import OLLAMA_EMBED_URL, OLLAMA_EMBEDDING_MODEL, OUT_PATH
from talk_rag_scheduler.indexing_pipeline import ingest_knowledge_and_run_indexing
from talk_rag_scheduler.models import IndexingKwargs, RetrievalKwargs, RetrievalResult
from talk_rag_scheduler.utils import prettify_document_info

RETRIEVAL_KEYWORD_PIPE_IMG = OUT_PATH / "retrieval_keyword_pipeline.png"
RETRIEVAL_VECTOR_PIPE_IMG = OUT_PATH / "retrieval_vector_pipeline.png"


def create_retrieval_keyword_pipeline(
    document_store: InMemoryDocumentStore,
) -> Pipeline:
    retrieval = Pipeline()
    retrieval.add_component("retriever", InMemoryBM25Retriever(document_store))
    OUT_PATH.mkdir(exist_ok=True)
    retrieval.draw(RETRIEVAL_KEYWORD_PIPE_IMG)
    return retrieval


def create_retrieval_embedding_pipeline(
    document_store: InMemoryDocumentStore,
    ollama_embedding_model: str = OLLAMA_EMBEDDING_MODEL,
) -> Pipeline:
    retrieval_embedding_pipeline = Pipeline()
    retrieval_embedding_pipeline.add_component(
        "text_embedder",
        OllamaTextEmbedder(model=ollama_embedding_model, url=OLLAMA_EMBED_URL),
    )
    retrieval_embedding_pipeline.add_component(
        "retriever", InMemoryEmbeddingRetriever(document_store)
    )
    retrieval_embedding_pipeline.connect(
        "text_embedder.embedding", "retriever.query_embedding"
    )
    OUT_PATH.mkdir(exist_ok=True)
    retrieval_embedding_pipeline.draw(RETRIEVAL_VECTOR_PIPE_IMG)
    return retrieval_embedding_pipeline


def main() -> None:
    parser = ArgumentParser(
        prog="Retrieval over the python scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Query the scheduler documents",
        epilog="""Example usage:
  python -m talk_rag_scheduler.retrieval_pipeline "What are the features of the scheduler?" --top_k 3
  python -m talk_rag_scheduler.retrieval_pipeline "What are the features of the scheduler?" --indexing_mode semantic
  python -m talk_rag_scheduler.retrieval_pipeline "What are the features of the scheduler?" --indexing_mode semantic_split --split_length 8
""",
    )
    add_retrieval_args(parser)
    indexing_kwargs: IndexingKwargs = parse_indexing_args(parser.parse_args())
    retrieval_kwargs: RetrievalKwargs = parse_retrieval_kwargs(parser.parse_args())

    document_store = InMemoryDocumentStore()
    ingest_knowledge_and_run_indexing(
        document_store,
        indexing_mode=indexing_kwargs["indexing_mode"],
        split_by=indexing_kwargs["split_by"],
        split_length=indexing_kwargs["split_length"],
        split_overlap=indexing_kwargs["split_overlap"],
        split_threshold=indexing_kwargs["split_threshold"],
        ollama_embedding_model=OLLAMA_EMBEDDING_MODEL,
    )

    results: RetrievalResult
    if retrieval_kwargs["indexing_mode"] == "keyword":
        retrieval_bm25_pipeline = create_retrieval_keyword_pipeline(document_store)
        results = retrieval_bm25_pipeline.run(
            {
                "query": retrieval_kwargs["query"],
                "top_k": retrieval_kwargs["top_k"],
            }
        )
    elif retrieval_kwargs["indexing_mode"] in ["semantic", "semantic_split"]:
        retrieval_embedding_pipeline = create_retrieval_embedding_pipeline(
            document_store, ollama_embedding_model=OLLAMA_EMBEDDING_MODEL
        )
        results = retrieval_embedding_pipeline.run(
            {
                "text": retrieval_kwargs["query"],
                "top_k": retrieval_kwargs["top_k"],
            }
        )

    for document in results["retriever"]["documents"]:
        print("```text")
        print(prettify_document_info(document))
        print("```")
        print("---")
        print(f"{document.content}\n\n")
        print("---")
        print("---")


if __name__ == "__main__":
    main()
