import argparse
import asyncio
from pathlib import Path

import dotenv

from dataset import Dataset
from evaluator import AsyncGeminiEvaluator
from logs import get_logger


dotenv.load_dotenv()
logger = get_logger(__name__)


def normalize_model_id(model: str) -> str:
    return model.replace("/", "-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation for inference results."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Model ID to evaluate. Can be either raw model name, "
            "e.g. 'gemini/gemini-3.5-flash', or normalized model_id, "
            "e.g. 'gemini-gemini-3.5-flash'."
        ),
    )

    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        help="Pipeline ID to evaluate.",
    )

    parser.add_argument(
        "--grader-model",
        type=str,
        default="gemini-2.5-flash",
        help="Model ID used as evaluator/grader.",
    )

    parser.add_argument(
        "--eval-dataset",
        type=Path,
        default=Path("data/eval_dataset.jsonl"),
        help="Path to dataset containing updated rubrics.",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Base directory containing inference results.",
    )

    parser.add_argument(
        "--evals-dir",
        type=Path,
        default=Path("evals"),
        help="Base directory where evaluation metrics will be saved.",
    )

    return parser.parse_args()


async def run_single_eval(
    model_id: str,
    pipeline_id: str,
    grader_model_id: str,
    eval_dataset_path: Path | None,
    results_base_path: Path,
    evals_base_path: Path,
) -> None:
    logger.info(f"Running {pipeline_id} on {model_id}...")

    evaluator = AsyncGeminiEvaluator(model_id=grader_model_id)

    dataset_path = results_base_path / pipeline_id / f"{model_id}.jsonl"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Inference results file not found: {dataset_path}")

    inference_results = Dataset(dataset_path)

    if eval_dataset_path and eval_dataset_path.exists():
        logger.info(f"Loaded rubrics from {eval_dataset_path}")
        inference_results = inference_results.get_rubrics_from_other_dataset(
            Dataset(eval_dataset_path)
        )

    metrics = await evaluator(inputs=inference_results)

    metrics.model_id = model_id
    metrics.eval_model_id = grader_model_id
    metrics.pipeline_id = pipeline_id

    output_path = evals_base_path / pipeline_id / f"{model_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics.save_to_disk(output_path)

    logger.info(f"Saved results to {output_path}")


async def main() -> None:
    args = parse_args()

    model_id = normalize_model_id(args.model)
    pipeline_id = args.pipeline

    args.evals_dir.mkdir(exist_ok=True, parents=True)

    await run_single_eval(
        model_id=model_id,
        pipeline_id=pipeline_id,
        grader_model_id=args.grader_model,
        eval_dataset_path=args.eval_dataset,
        results_base_path=args.results_dir,
        evals_base_path=args.evals_dir,
    )


if __name__ == "__main__":
    asyncio.run(main())
    logger.info("Done!")
