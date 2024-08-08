import argparse
import datetime as dt
from collections.abc import Sequence

from haystack.dataclasses.document import Document

from talk_rag_scheduler.const import (
    SPLIT_BY,
    SPLIT_LENGTH,
    SPLIT_OVERLAP,
    SPLIT_THRESHOLD,
)
from talk_rag_scheduler.models import ConcurrentRagOutput, MetaType


def calc_response_tps(meta: MetaType) -> float:
    return meta["eval_count"] / (meta["eval_duration"] / 1e9)


def calc_total_response_tps(meta: MetaType) -> float:
    return meta["eval_count"] / (meta["total_duration"] / 1e9)


def calc_prompt_tps(meta: MetaType) -> float:
    return meta["prompt_eval_count"] / (meta["prompt_eval_duration"] / 1e9)


def created_at_to_datetime(created_at: str) -> dt.datetime:
    # created_at has the format 2024-08-04T03:23:37.117581972Z
    return dt.datetime.fromisoformat(created_at)


def list_evaluation_counts(metas: Sequence[MetaType]) -> list[int]:
    return [x["eval_count"] for x in metas]


def list_prompt_evaluation_counts(metas: Sequence[MetaType]) -> list[int]:
    return [x["prompt_eval_count"] for x in metas]


def list_concurrent_eval_count(results: ConcurrentRagOutput) -> list[int]:
    return [sum(list_evaluation_counts(x["meta"])) for x in results["llms"].values()]


def list_concurrent_prompt_eval_count(results: ConcurrentRagOutput) -> list[int]:
    return [
        sum(list_prompt_evaluation_counts(x["meta"])) for x in results["llms"].values()
    ]


def calc_concurrent_e2e_throughput(
    results: ConcurrentRagOutput, t_total: float
) -> float:
    concurrent_eval_count = list_concurrent_eval_count(results)
    return sum(concurrent_eval_count) / t_total


def prettify_document_info(doc: Document) -> str:
    document_info = f"""Score = {doc.score}
Document ID = {doc.id}
Document file path = {doc.meta['file_path']}
Content Type = {doc.content_type}
Embedding = {doc.embedding}"""
    return document_info


def parse_indexing_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Indexing",
        description="Index the scheduler documents",
    )
    parser.add_argument(
        "--indexing_mode",
        choices=["simple", "semantic", "semantic_split"],
        default=None,
    )
    parser.add_argument(
        "--split_by", choices=["word", "sentence", "page", "passage"], default=SPLIT_BY
    )
    parser.add_argument("--split_length", type=int, default=SPLIT_LENGTH)
    parser.add_argument("--split_overlap", type=int, default=SPLIT_OVERLAP)
    parser.add_argument("--split_threshold", type=int, default=SPLIT_THRESHOLD)

    return parser.parse_args()


def main() -> None:
    namespace = parse_indexing_args()
    print(namespace)


if __name__ == "__main__":
    main()
