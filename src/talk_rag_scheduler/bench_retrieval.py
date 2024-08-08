from pathlib import Path
from typing import Literal
from argparse import ArgumentParser
import argparse

from haystack.dataclasses.document import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from talk_rag_scheduler.const import GOLDEN_RETRIEVALS, SCHEDULER_SRC_PATH, TOP_K
from talk_rag_scheduler.indexing_pipeline import (
    ingest_knowledge_and_run_indexing,
    remote_files_generator,
)
from talk_rag_scheduler.models import ExtendedConfusionMatrix, RetrievalResult
from talk_rag_scheduler.retrieval_pipeline import (
    create_retrieval_embedding_pipeline,
    create_retrieval_keyword_pipeline,
)


def get_relative_dir_from_documents(documents: list[Document]) -> list[str]:
    return [
        str(Path(doc.meta["file_path"]).relative_to(SCHEDULER_SRC_PATH))
        for doc in documents
    ]


def check_true_positive(
    results: RetrievalResult, file_paths_actual_positive: list[str]
) -> list[bool]:
    documents_predicted_positive = results["retriever"]["documents"]

    is_true_positive = [
        dir in file_paths_actual_positive
        for dir in get_relative_dir_from_documents(documents_predicted_positive)
    ]
    return is_true_positive


def calc_ext_confusion_matrix(
    results: RetrievalResult, file_paths_actual_positive: list[str], n_documents: int
) -> ExtendedConfusionMatrix:
    documents_predicted_positive = results["retriever"]["documents"]
    is_true_positive = check_true_positive(results, file_paths_actual_positive)

    TP = sum(is_true_positive)
    FP = len(documents_predicted_positive) - TP
    FN = len(file_paths_actual_positive) - TP
    TN = n_documents - (TP + FP + FN)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    F1 = 2 * (TP / (2 * TP + FP + FN))
    TPR = TP / (TP + FN)
    PPV = TP / (TP + FP)

    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "ACC": ACC,
        "F1": F1,
        "TPR": TPR,
        "PPV": PPV,
    }


def pretty_confusion_info(cm: ExtendedConfusionMatrix) -> str:
    return f"""TP = {cm['TP']}
FP = {cm['FP']}
TN = {cm['TN']}
FN = {cm['FN']}"""


def pretty_scores_info(cm: ExtendedConfusionMatrix, top_k: int) -> str:
    return f"""Accuracy@{top_k} = {cm['ACC']}
F1 Score@{top_k} = {cm['F1']}
Recall@{top_k} = {cm['TPR']}
Precision@{top_k} = {cm['PPV']}"""


def bench_bm25_retrieval(
    top_k: int = TOP_K, golden_id: int = 0
) -> tuple[list[bool], ExtendedConfusionMatrix]:
    document_store = InMemoryDocumentStore()
    ingest_knowledge_and_run_indexing(document_store)

    source_files = list(remote_files_generator())
    n_documents = len(source_files)

    query = GOLDEN_RETRIEVALS[golden_id]["query"]
    retrieval_bm25_pipeline = create_retrieval_keyword_pipeline(document_store)
    results: RetrievalResult = retrieval_bm25_pipeline.run(
        {"query": query, "top_k": top_k}
    )

    file_paths_actual_positive = GOLDEN_RETRIEVALS[golden_id]["golden_file_paths"]
    is_true_positive = check_true_positive(results, file_paths_actual_positive)

    confusion_matrix: ExtendedConfusionMatrix = calc_ext_confusion_matrix(
        results=results,
        file_paths_actual_positive=file_paths_actual_positive,
        n_documents=n_documents,
    )
    return is_true_positive, confusion_matrix


def bench_vector_retrieval(
    top_k: int = TOP_K,
    mode: Literal["semantic", "semantic_split"] = "semantic",
    golden_id: int = 0,
) -> tuple[list[bool], ExtendedConfusionMatrix]:
    document_store = InMemoryDocumentStore()
    ingest_knowledge_and_run_indexing(
        document_store, indexing_mode=mode, ollama_embedding_model="mxbai-embed-large"
    )
    source_files = list(remote_files_generator())
    n_documents = len(source_files)

    query = GOLDEN_RETRIEVALS[golden_id]["query"]

    retrieval_embedding_pipeline = create_retrieval_embedding_pipeline(
        document_store, ollama_embedding_model="mxbai-embed-large"
    )
    results = retrieval_embedding_pipeline.run({"text": query, "top_k": top_k})

    file_paths_actual_positive = GOLDEN_RETRIEVALS[golden_id]["golden_file_paths"]
    is_true_positive = check_true_positive(results, file_paths_actual_positive)

    confusion_matrix: ExtendedConfusionMatrix = calc_ext_confusion_matrix(
        results=results,
        file_paths_actual_positive=file_paths_actual_positive,
        n_documents=n_documents,
    )
    return is_true_positive, confusion_matrix


def main() -> None:
    parser = ArgumentParser(
        prog="Rudimentary benchmarking of retrieval pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Query the scheduler documents",
        epilog="""Example usage:
  python -m talk_rag_scheduler.bench_retrieval --top_k 3
""",
    )
    # TODO: use modular approach as for the other files using argument_parser.py
    # and allow the additional options
    parser.add_argument(
        "--top_k",
        type=int,
        default=TOP_K,
        help=f"Number of documents to retrieve, defaults to {TOP_K}",
    )
    top_k = parser.parse_args().top_k

    is_true_positive, confusion_matrix = bench_bm25_retrieval(top_k=top_k)
    print(is_true_positive)
    print(pretty_confusion_info(confusion_matrix))
    print(pretty_scores_info(confusion_matrix, top_k=top_k))

    is_true_positive, confusion_matrix = bench_vector_retrieval(top_k=top_k)
    print(is_true_positive)
    print(pretty_confusion_info(confusion_matrix))
    print(pretty_scores_info(confusion_matrix, top_k=top_k))

    is_true_positive, confusion_matrix = bench_vector_retrieval(
        top_k=top_k, mode="semantic_split"
    )
    print(is_true_positive)
    print(pretty_confusion_info(confusion_matrix))
    print(pretty_scores_info(confusion_matrix, top_k=top_k))


if __name__ == "__main__":
    main()
