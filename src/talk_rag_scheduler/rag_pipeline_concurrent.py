import argparse
import asyncio
import math
import time
from argparse import ArgumentParser
from typing import Any, assert_never

from haystack import Pipeline, component, default_from_dict, default_to_dict
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.core.component import Component
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils.type_serialization import deserialize_type
from haystack_integrations.components.generators.ollama import OllamaGenerator

from talk_rag_scheduler.argument_parser import (
    add_concurrent_pipeline_args,
    parse_concurrent_pipeline_args,
)
from talk_rag_scheduler.const import (
    INDEXING_MODE,
    N_CONCURRENT,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_GENERATE_URL,
    OLLAMA_MODEL,
    OUT_PATH,
    PROMPT_TEMPLATE,
)
from talk_rag_scheduler.indexing_pipeline import ingest_knowledge_and_run_indexing
from talk_rag_scheduler.models import (
    T_INDEXING_MODE,
    ConcurrentRagOutput,
    ConcurrentRagPipelineKwargs,
)
from talk_rag_scheduler.retrieval_pipeline import (
    create_retrieval_embedding_pipeline,
    create_retrieval_keyword_pipeline,
)
from talk_rag_scheduler.utils import list_concurrent_eval_count

ASYNCIO_PATCHED: bool = False
CONCURRENT_KEYWORD_RAG_PIPELINE_IMG = OUT_PATH / "concurrent_keyword_rag_pipeline.png"
CONCURRENT_SEMANTIC_RAG_PIPELINE_IMG = OUT_PATH / "concurrent_semantic_rag_pipeline.png"


def is_notebook() -> bool:
    try:
        shell: str = get_ipython().__class__.__name__  # type: ignore
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False


def patch_asyncio() -> None:
    """Jupyter Notebooks already runs in an event loop. Patch asyncio to allow nesting."""
    global ASYNCIO_PATCHED
    if not ASYNCIO_PATCHED and is_notebook():
        import nest_asyncio

        nest_asyncio.apply()
        ASYNCIO_PATCHED = True


@component
class ConcurrentGenerators:
    def __init__(
        self,
        generators: list[Component],
        names: list[str] | None = None,
    ):
        patch_asyncio()
        self.generators = generators
        if names is None:
            names = [f"generator_{i}" for i in range(len(generators))]
        self.names = names

        # we set the output types here so that the results are not too nested
        output_types = {k: dict[str, Any] for k in names}
        component.set_output_types(self, **output_types)

    def warm_up(self) -> None:
        """Warm up the generators."""
        for generator in self.generators:
            if hasattr(generator, "warm_up"):
                generator.warm_up()

    async def _arun(self, **kwargs: Any) -> dict[str, dict[str, Any]]:
        """
        Asynchrounous method to run the generators concurrently.
        """

        # the generators run in separate threads
        results = await asyncio.gather(
            *[
                asyncio.to_thread(generator.run, **kwargs)
                for generator in self.generators
            ]
        )

        organized_results = {}
        for generator_name, res_ in zip(self.names, results):
            organized_results[generator_name] = res_
        return organized_results

    def run(
        self,
        prompt: str,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Synchronous run method that can be integrated into a classic synchronous pipeline.
        """
        if generation_kwargs is None:
            generation_kwargs = {}
        task = self._arun(prompt=prompt, generation_kwargs=generation_kwargs)
        results = asyncio.new_event_loop().run_until_complete(task)

        return results

    def to_dict(self) -> dict[str, Any]:
        generators = [generator.to_dict() for generator in self.generators]
        res: dict[str, Any] = default_to_dict(
            self, generators=generators, names=self.names
        )
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConcurrentGenerators":
        init_params = data.get("init_parameters", {})

        # Deserialize the generators
        generators = []
        serialized_generators = init_params["generators"]
        for serialized_generator in serialized_generators:
            generator_class = deserialize_type(serialized_generator["type"])
            generator = generator_class.from_dict(serialized_generator)
            generators.append(generator)

        data["init_parameters"]["generators"] = generators
        res: ConcurrentGenerators = default_from_dict(cls, data)
        return res


def create_concurrent_rag_pipeline(
    document_store: InMemoryDocumentStore,
    n_concurrent: int = N_CONCURRENT,
    ollama_model: str = OLLAMA_MODEL,
    indexing_mode: T_INDEXING_MODE = INDEXING_MODE,
    ollama_embedding_model: str = OLLAMA_EMBEDDING_MODEL,
) -> Pipeline:
    prompt_builder = PromptBuilder(template=PROMPT_TEMPLATE)
    llm = OllamaGenerator(
        model=ollama_model,
        url=OLLAMA_GENERATE_URL,
        generation_kwargs={
            # "num_predict": 100,
            # "temperature": 0.9,
        },
    )

    if indexing_mode == "keyword":
        rag_pipeline = create_retrieval_keyword_pipeline(document_store)
    elif indexing_mode in ["semantic", "semantic_split"]:
        rag_pipeline = create_retrieval_embedding_pipeline(
            document_store, ollama_embedding_model=ollama_embedding_model
        )

    rag_pipeline.add_component("prompt_builder", prompt_builder)
    n_pad = math.ceil(math.log10(n_concurrent))
    rag_pipeline.add_component(
        "llms",
        ConcurrentGenerators(
            generators=[llm] * n_concurrent,
            names=[f"{llm.model}_{x:_>{n_pad}}" for x in range(n_concurrent)],
        ),
    )
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llms")
    match indexing_mode:
        case "keyword":
            rag_pipeline.draw(CONCURRENT_KEYWORD_RAG_PIPELINE_IMG)
        case "semantic" | "semantic_split":
            rag_pipeline.draw(CONCURRENT_SEMANTIC_RAG_PIPELINE_IMG)
        case _ as unreachable:
            assert_never(unreachable)
            raise ValueError("Unexpected Arguments")

    return rag_pipeline


def list_concurrent_replies(results: ConcurrentRagOutput) -> list[list[str]]:
    replies = [x["replies"] for x in results["llms"].values()]
    return replies


def prettify_concurrent_rag_info(results: ConcurrentRagOutput) -> str:
    tokens_per_response = list_concurrent_eval_count(results)
    s = f"""Tokens per response: {tokens_per_response}
Total tokens generated: {sum(tokens_per_response)}
Total time taken: {results["t_total"]:.4} s
E2e throughput: {sum(tokens_per_response) / results["t_total"]:.2f} token/s"""
    return s


def main() -> None:
    parser = ArgumentParser(
        prog="Concurrent Rag Pipeline over the python scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Query the scheduler documents",
        epilog="""Example usage:
  python -m talk_rag_scheduler.rag_pipeline_concurrent "What are the features of the scheduler?" --n_concurrent 5
""",
    )
    add_concurrent_pipeline_args(parser)
    concurrent_pipeline_kwargs: ConcurrentRagPipelineKwargs = (
        parse_concurrent_pipeline_args(parser.parse_args())
    )

    document_store = InMemoryDocumentStore()
    ingest_knowledge_and_run_indexing(
        document_store,
        indexing_mode=concurrent_pipeline_kwargs["indexing_mode"],
        split_by=concurrent_pipeline_kwargs["split_by"],
        split_length=concurrent_pipeline_kwargs["split_length"],
        split_overlap=concurrent_pipeline_kwargs["split_overlap"],
        split_threshold=concurrent_pipeline_kwargs["split_threshold"],
        ollama_embedding_model=concurrent_pipeline_kwargs["ollama_embedding_model"],
    )
    concurrent_rag_pipeline: Pipeline = create_concurrent_rag_pipeline(
        document_store=document_store,
        n_concurrent=concurrent_pipeline_kwargs["n_concurrent"],
        ollama_model=concurrent_pipeline_kwargs["ollama_model"],
    )
    t_start = time.perf_counter()

    if concurrent_pipeline_kwargs["indexing_mode"] == "keyword":
        retrieval_kwargs = {
            "retriever": {
                "query": concurrent_pipeline_kwargs["query"],
                "top_k": concurrent_pipeline_kwargs["top_k"],
            },
        }
    elif concurrent_pipeline_kwargs["indexing_mode"] in ["semantic", "semantic_split"]:
        retrieval_kwargs = {
            "text_embedder": {"text": concurrent_pipeline_kwargs["query"]},
            "retriever": {
                "top_k": concurrent_pipeline_kwargs["top_k"],
            },
        }
    generator_kwargs = {
        "prompt_builder": {"query": concurrent_pipeline_kwargs["query"]},
        "llms": {
            "generation_kwargs": {
                "num_predict": concurrent_pipeline_kwargs["num_predict"],
                "temperature": concurrent_pipeline_kwargs["temperature"],
            },
        },
    }

    results = concurrent_rag_pipeline.run({**generator_kwargs, **retrieval_kwargs})
    t_total = time.perf_counter() - t_start
    results["t_total"] = t_total

    for llm_res in list_concurrent_replies(results):
        print(llm_res[0])
        print("\n---")
    print("---")
    print("```text")
    print(prettify_concurrent_rag_info(results))
    print("```")
    print("---")
    print("---")


if __name__ == "__main__":
    main()
