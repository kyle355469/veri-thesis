from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import iter_jsonl_documents
from .embeddings import make_embedder
from .evaluation import run_evaluation
from .pipeline import RagRtlPipeline
from .types import RtlTask
from .vector_store import VectorStore, build_vector_store


def cmd_index(args: argparse.Namespace) -> None:
    embedder = make_embedder(args.embedder)
    documents = list(iter_jsonl_documents(args.corpus, limit=args.limit))
    texts = [document.retrieval_text for document in documents]
    vectors = embedder.encode(texts)
    store = build_vector_store(documents, vectors)
    store.save(args.output)
    print(f"Indexed {len(documents)} documents into {args.output}")


def cmd_generate(args: argparse.Namespace) -> None:
    embedder = make_embedder(args.embedder)
    store = VectorStore.load(args.index)
    pipeline = RagRtlPipeline(
        store=store,
        embedder=embedder,
        cache_path=args.cache,
        monitor_path=args.monitor,
    )
    task = RtlTask(
        prompt=args.prompt or Path(args.prompt_file).read_text(encoding="utf-8"),
        target_hdl=args.target_hdl,
        module_signature=args.module_signature,
        constraints=args.constraint,
        max_repair_attempts=args.max_repair_attempts,
    )
    response = pipeline.run(task, retrieve_k=args.retrieve_k, context_k=args.context_k)
    print(response.rtl)
    if args.json_report:
        Path(args.json_report).write_text(json.dumps(response, default=lambda value: getattr(value, "__dict__", str(value)), indent=2), encoding="utf-8")


def cmd_evaluate(args: argparse.Namespace) -> None:
    embedder = make_embedder(args.embedder)
    store = VectorStore.load(args.index)
    summary = run_evaluation(
        tasks_path=args.tasks,
        store=store,
        embedder=embedder,
        mode=args.mode,
        output_path=args.output,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG-assisted RTL generation prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="Build a vector index from merged.jsonl")
    index.add_argument("--corpus", default="merged.jsonl")
    index.add_argument("--output", default="indexes/rtl_hash")
    index.add_argument("--embedder", default="hash")
    index.add_argument("--limit", type=int, default=None)
    index.set_defaults(func=cmd_index)

    generate = subparsers.add_parser("generate", help="Generate RTL with retrieval and verification feedback")
    generate.add_argument("--index", default="indexes/rtl_hash")
    generate.add_argument("--embedder", default="hash")
    generate.add_argument("--prompt")
    generate.add_argument("--prompt-file")
    generate.add_argument("--target-hdl", default="verilog")
    generate.add_argument("--module-signature")
    generate.add_argument("--constraint", action="append", default=[])
    generate.add_argument("--retrieve-k", type=int, default=8)
    generate.add_argument("--context-k", type=int, default=4)
    generate.add_argument("--max-repair-attempts", type=int, default=1)
    generate.add_argument("--cache", default="data/history_cache.json")
    generate.add_argument("--monitor", default="runs/monitor.jsonl")
    generate.add_argument("--json-report")
    generate.set_defaults(func=cmd_generate)

    evaluate = subparsers.add_parser("evaluate", help="Run thesis baseline evaluation on a JSONL prompt set")
    evaluate.add_argument("--tasks", required=True, help="JSONL with at least a 'prompt' field per row")
    evaluate.add_argument("--index", default="indexes/rtl_hash")
    evaluate.add_argument("--embedder", default="hash")
    evaluate.add_argument("--mode", choices=["llm_only", "rag", "rag_cache_verify"], default="rag_cache_verify")
    evaluate.add_argument("--output", default="runs/evaluation.json")
    evaluate.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "command", None) == "generate" and not args.prompt and not args.prompt_file:
        parser.error("generate requires --prompt or --prompt-file")
    args.func(args)


if __name__ == "__main__":
    main()
