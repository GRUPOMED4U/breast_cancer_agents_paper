import argparse
import asyncio
from pathlib import Path

import dotenv
import jsonlines
import yaml

from dataset import Dataset
from pipelines import PIPELINE_REGISTRY
from logs import get_logger


dotenv.load_dotenv()
logger = get_logger(__name__, level="DEBUG")


def normalize_model_id(model: str) -> str:
    return model.replace("/", "-").replace(":", "-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference pipeline.")

    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=PIPELINE_REGISTRY.keys(),
        help="Pipeline ID to run.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Model ID to run. Can be either the raw model name, "
            "e.g. 'gemini/gemini-2.5-flash', or normalized model_id, "
            "e.g. 'gemini-gemini-2.5-flash'."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval_dataset.jsonl"),
        help="Path to evaluation dataset JSONL file.",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Base directory where results will be saved.",
    )

    parser.add_argument(
        "--save-steps",
        type=int,
        default=1,
        help="Number of examples to process before saving results.",
    )

    parser.add_argument(
        "--limit-to",
        type=int,
        default=None,
        help="Limit processing to the first N examples.",
    )

    parser.add_argument(
        "-t",
        "--tag",
        type=str,
        default=None,
        help="Optional tag to categorize results.",
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def experiment_matches_model(exp_config: dict, requested_model: str) -> bool:
    agent_definition = exp_config.get("agent_definition", {})
    model = agent_definition.get("model")

    if model is None:
        return False

    return requested_model in {
        model,
        normalize_model_id(model),
    }


async def main() -> None:
    args = parse_args()

    results_base_path = args.results_dir
    results_base_path.mkdir(exist_ok=True, parents=True)

    eval_dataset = Dataset(args.dataset)

    pipeline = args.pipeline
    requested_model = args.model

    pipeline_exp_path = Path(f"experiments/{pipeline}.yaml")
    pipeline_exp_config = load_yaml(pipeline_exp_path)

    experiments = pipeline_exp_config["experiments"]
    pipeline_class = PIPELINE_REGISTRY[pipeline]

    matching_experiments = [
        exp_config
        for exp_config in experiments
        if experiment_matches_model(exp_config, requested_model)
    ]

    if not matching_experiments:
        available_models = sorted(
            {
                exp_config["agent_definition"]["model"]
                for exp_config in experiments
                if "agent_definition" in exp_config
                and "model" in exp_config["agent_definition"]
            }
        )

        raise ValueError(
            f"No experiment found for model '{requested_model}' "
            f"in pipeline '{pipeline}'.\n"
            f"Available models:\n"
            + "\n".join(f"- {model}" for model in available_models)
        )

    for exp_config in matching_experiments:
        pipeline_instance = pipeline_class(**exp_config)

        logger.debug(f"Experiment config: {exp_config}")

        results_path = (
            results_base_path
            / pipeline
            / f"{pipeline_instance.model_id}{'_' + args.tag.lower().replace(' ', '_') if args.tag else ''}.jsonl"
        )

        dataset_for_inference = eval_dataset

        if results_path.exists():
            logger.info(
                f"File related to {pipeline}/{pipeline_instance.model_id} already exists."
            )
            logger.info("Loading existing results...")

            previous_results = Dataset(results_path)
            dataset_for_inference = dataset_for_inference.merge(previous_results)

        logger.info(
            f"Running inference related to {pipeline}/{pipeline_instance.model_id}."
        )

        if args.limit_to:
            dataset_for_inference = dataset_for_inference[: args.limit_to]

        for idx in range(0, len(dataset_for_inference), args.save_steps):
            batch = dataset_for_inference[idx : idx + args.save_steps]
            results = await pipeline_instance.run_pipeline(batch)

            results_path.parent.mkdir(exist_ok=True, parents=True)

            mode = "w" if idx == 0 else "a"
            with jsonlines.open(results_path, mode) as writer:
                writer.write_all([r.model_dump() for r in results])


if __name__ == "__main__":
    asyncio.run(main())
