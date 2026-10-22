import argparse
import time

from contract_review.audit import task as audit
from contract_review.classification import task as classification
from contract_review.data import task as data
from contract_review.evaluation import task as evaluation
from contract_review.retrieval import task as retrieval
from contract_review.utils import make_context, write_run_info

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistant de revue de contrats")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="lance tout ou partie du pipeline")
    run_parser.add_argument("--exp", required=True, help="nom dans configs/experiments/")
    run_parser.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS))
    run_parser.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
