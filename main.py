import argparse
import json
import time

from contract_review.audit import task as audit
from contract_review.classification import task as classification
from contract_review.data import task as data
from contract_review.evaluation import task as evaluation
from contract_review.inference import ClausePredictor, ClauseSearcher
from contract_review.retrieval import task as retrieval
from contract_review.utils import find_run, make_context, write_run_info

# Ordre du pipeline. Chaque étape est une fonction run(ctx).
STEPS = {
    "data": data.run,
    "classification": classification.run,
    "evaluation": evaluation.run,
    "retrieval": retrieval.run,
    "audit": audit.run,
}


def run(args: argparse.Namespace) -> None:
    ctx = make_context(args.exp)
    durations = {}
    for name, step in STEPS.items():
        if name in args.steps:
            ctx.log.info("étape %s", name)
            start = time.perf_counter()
            step(ctx)
            durations[name] = round(time.perf_counter() - start, 1)
    write_run_info(ctx, durations)


def predict(args: argparse.Namespace) -> None:
    result = ClausePredictor(find_run(args.exp)).predict([args.text])[0]
    print(json.dumps(result, indent=2, ensure_ascii=False))


def search(args: argparse.Namespace) -> None:
    run_dir = find_run(args.exp)
    searcher = ClauseSearcher(run_dir, ClausePredictor(run_dir))
    find = searcher.search_clause if args.clause else searcher.search_question
    for rank, hit in enumerate(find(args.text, args.k), start=1):
        print(f"{rank}. [{hit['type']}] (score {hit['score']:.3f}) {hit['text'][:300]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistant de revue de contrats")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="lance tout ou partie du pipeline")
    run_parser.add_argument("--exp", required=True, help="nom dans configs/experiments/")
    run_parser.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS))
    run_parser.set_defaults(func=run)

    predict_parser = commands.add_parser("predict", help="type, confiance et revue d'une clause")
    predict_parser.add_argument("text", help="la clause")
    predict_parser.set_defaults(func=predict)

    search_parser = commands.add_parser("search", help="cherche des clauses similaires")
    search_parser.add_argument("text", help="une question de juriste (ou une clause avec --clause)")
    search_parser.add_argument("--k", type=int, default=5)
    search_parser.add_argument("--clause", action="store_true", help="le texte est une clause")
    search_parser.set_defaults(func=search)

    for parser_with_exp in (predict_parser, search_parser):
        parser_with_exp.add_argument(
            "--exp", default="legalbert", help="expérience du run à utiliser"
        )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
