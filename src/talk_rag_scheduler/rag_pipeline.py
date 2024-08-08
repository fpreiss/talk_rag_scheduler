import argparse
from argparse import ArgumentParser
from typing import assert_never

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.generators.ollama import OllamaGenerator

from talk_rag_scheduler.argument_parser import add_pipeline_args, parse_pipeline_args
from talk_rag_scheduler.const import (
    INDEXING_MODE,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_GENERATE_URL,
    OLLAMA_MODEL,
    OUT_PATH,
    PROMPT_TEMPLATE,
)
from talk_rag_scheduler.indexing_pipeline import ingest_knowledge_and_run_indexing
from talk_rag_scheduler.models import (
    T_INDEXING_MODE,
    MetaType,
    RagOutput,
    RagPipelineKwargs,
)
from talk_rag_scheduler.retrieval_pipeline import (
    create_retrieval_embedding_pipeline,
    create_retrieval_keyword_pipeline,
)
from talk_rag_scheduler.utils import (
    calc_prompt_tps,
    calc_response_tps,
    calc_total_response_tps,
)

KEYWORD_RAG_PIPELINE_IMG = OUT_PATH / "keyword_rag_pipeline.png"
SEMANTIC_RAG_PIPELINE_IMG = OUT_PATH / "semantic_rag_pipeline.png"


def create_rag_pipeline(
    document_store: InMemoryDocumentStore,
    ollama_model: str = OLLAMA_MODEL,
    ollama_embedding_model: str = OLLAMA_EMBEDDING_MODEL,
    indexing_mode: T_INDEXING_MODE = INDEXING_MODE,
) -> Pipeline:
    prompt_builder = PromptBuilder(template=PROMPT_TEMPLATE)
    llm = OllamaGenerator(
        model=ollama_model,
        url=OLLAMA_GENERATE_URL,
        generation_kwargs={
            # "num_predict": NUM_PREDICT,
            # "temperature": TEMPERATURE,
        },
    )

    if indexing_mode == "keyword":
        rag_pipeline = create_retrieval_keyword_pipeline(document_store)
    elif indexing_mode in ["semantic", "semantic_split"]:
        rag_pipeline = create_retrieval_embedding_pipeline(
            document_store, ollama_embedding_model=ollama_embedding_model
        )

    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", llm)
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")

    match indexing_mode:
        case "keyword":
            rag_pipeline.draw(KEYWORD_RAG_PIPELINE_IMG)
        case "semantic" | "semantic_split":
            rag_pipeline.draw(SEMANTIC_RAG_PIPELINE_IMG)
        case _ as unreachable:
            assert_never(unreachable)
            raise ValueError("Unexpected Arguments")

    return rag_pipeline


def prettify_rag_info(meta: MetaType) -> str:
    s = f"""prompt tokens per second = {calc_prompt_tps(meta)}
response tokens per second = {calc_response_tps(meta)}
total tokens per second = {calc_total_response_tps(meta)}"""
    return s


def main() -> None:
    parser = ArgumentParser(
        prog="Rag Pipeline over the python scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Query the scheduler documents",
        epilog="""Example usage:
  python -m talk_rag_scheduler.rag_pipeline "What are the features of the scheduler?" --num_predict 100
""",
    )
    add_pipeline_args(parser)
    pipeline_kwargs: RagPipelineKwargs = parse_pipeline_args(parser.parse_args())
    document_store = InMemoryDocumentStore()
    ingest_knowledge_and_run_indexing(
        document_store,
        indexing_mode=pipeline_kwargs["indexing_mode"],
        split_by=pipeline_kwargs["split_by"],
        split_length=pipeline_kwargs["split_length"],
        split_overlap=pipeline_kwargs["split_overlap"],
        split_threshold=pipeline_kwargs["split_threshold"],
        ollama_embedding_model=pipeline_kwargs["ollama_embedding_model"],
    )

    rag_pipeline: Pipeline = create_rag_pipeline(
        document_store=document_store,
        indexing_mode=pipeline_kwargs["indexing_mode"],
        ollama_model=pipeline_kwargs["ollama_model"],
        ollama_embedding_model=pipeline_kwargs["ollama_embedding_model"],
    )

    if pipeline_kwargs["indexing_mode"] == "keyword":
        retrieval_kwargs = {
            "retriever": {
                "query": pipeline_kwargs["query"],
                "top_k": pipeline_kwargs["top_k"],
            },
        }
    elif pipeline_kwargs["indexing_mode"] in ["semantic", "semantic_split"]:
        retrieval_kwargs = {
            "text_embedder": {"text": pipeline_kwargs["query"]},
            "retriever": {
                "top_k": pipeline_kwargs["top_k"],
            },
        }

    generator_kwargs = {
        "prompt_builder": {"query": pipeline_kwargs["query"]},
        "llm": {
            "generation_kwargs": {
                "num_predict": pipeline_kwargs["num_predict"],
                "temperature": pipeline_kwargs["temperature"],
            },
        },
    }

    results: RagOutput = rag_pipeline.run({**generator_kwargs, **retrieval_kwargs})
    reply: str = results["llm"]["replies"][0]
    meta: MetaType = results["llm"]["meta"][0]
    print(reply)
    print("\n---")
    print("```text")
    print(prettify_rag_info(meta))
    print("```")
    print("---")
    print("---")


if __name__ == "__main__":
    main()
