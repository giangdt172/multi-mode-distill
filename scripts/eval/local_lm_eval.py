#!/usr/bin/env python3
"""Run lm-evaluation-harness with datasets mirrored under EVAL_DATA_DIR."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

import datasets
import evaluate


DATASET_DIRECTORIES = {
    "openai/gsm8k": "gsm8k",
    "qintongli/GSM-Plus": "gsm_plus",
    "EleutherAI/hendrycks_math": "hendrycks_math",
    "google-research-datasets/mbpp": "mbpp",
    "allenai/sciq": "sciq",
    "cais/mmlu": "mmlu",
    "TIGER-Lab/MMLU-Pro": "mmlu_pro",
    "SaylorTwift/bbh": "bbh",
}

TASK_DATASETS = {
    "gsm8k_cot": {"openai/gsm8k"},
    "gsm_plus": {"qintongli/GSM-Plus"},
    "minerva_math": {"EleutherAI/hendrycks_math"},
    "mbpp": {"google-research-datasets/mbpp"},
    "mbpp_instruct": {"google-research-datasets/mbpp"},
    "sciq": {"allenai/sciq"},
    "mmlu_stem": {"cais/mmlu"},
    "mmlu_pro_math": {"TIGER-Lab/MMLU-Pro"},
    "bbh_cot_fewshot": {"SaylorTwift/bbh"},
}

DATA_FILE_PATTERN = re.compile(
    r"^(?P<split>.+?)(?:-\d+-of-\d+)?\.(?:arrow|csv|json|jsonl|parquet)$"
)


def eval_data_root() -> Path:
    configured = os.environ.get("EVAL_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "data" / "eval"


def local_dataset_path(repo_id: str) -> Path:
    return eval_data_root() / DATASET_DIRECTORIES[repo_id]


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            "Download the assets declared in download.txt before submitting the job."
        )


def local_data_files(config_path: Path) -> dict[str, list[str]]:
    """Keep non-standard splits such as MMLU's `dev` when loading a folder."""
    data_files: dict[str, list[str]] = {}
    for candidate in config_path.rglob("*"):
        if not candidate.is_file():
            continue
        match = DATA_FILE_PATTERN.match(candidate.name)
        if match:
            data_files.setdefault(match.group("split"), []).append(str(candidate))
    return data_files


def install_offline_loaders() -> None:
    original_load_dataset = datasets.load_dataset
    original_evaluate_load = evaluate.load

    def load_local_dataset(*args: Any, **kwargs: Any):
        source = args[0] if args else kwargs.get("path")
        if source not in DATASET_DIRECTORIES:
            return original_load_dataset(*args, **kwargs)

        local_path = local_dataset_path(str(source))
        require_path(local_path, f"dataset mirror for {source}")

        name = args[1] if len(args) > 1 else kwargs.get("name")
        config_path = local_path / str(name) if name else None
        if name is None and (local_path / "data").is_dir():
            config_path = local_path / "data"
        if config_path is not None and config_path.is_dir():
            # HF dataset snapshots store many named configs as subdirectories.
            # Loading that directory directly avoids resolving the repo config
            # through the Hub while preserving the task's requested split.
            local_path = config_path
            if len(args) > 1:
                mutable_args = list(args)
                mutable_args[1] = None
                args = tuple(mutable_args)
            elif name is not None:
                kwargs["name"] = None
            data_files = local_data_files(config_path)
            if data_files:
                kwargs["data_files"] = data_files

        if args:
            mutable_args = list(args)
            mutable_args[0] = str(local_path)
            args = tuple(mutable_args)
        else:
            kwargs["path"] = str(local_path)

        # Hub-only options do not apply when the first argument is a local path.
        kwargs.pop("revision", None)
        kwargs.pop("token", None)
        kwargs.pop("trust_remote_code", None)
        if os.environ.get("EVAL_DEBUG_LOCAL") == "1":
            print(
                f"local dataset: {source} (config={name}) -> {local_path}",
                file=sys.stderr,
            )
        return original_load_dataset(*args, **kwargs)

    def load_local_metric(path: str, *args: Any, **kwargs: Any):
        if path != "code_eval":
            return original_evaluate_load(path, *args, **kwargs)

        metric_file = eval_data_root() / "code_eval" / "code_eval.py"
        require_path(metric_file, "local code_eval metric")
        require_path(metric_file.with_name("execute.py"), "local code_eval executor")
        return original_evaluate_load(str(metric_file), *args, **kwargs)

    datasets.load_dataset = load_local_dataset
    evaluate.load = load_local_metric


def datasets_for_tasks(tasks: list[str]) -> set[str]:
    required: set[str] = set()
    for task in tasks:
        required.update(TASK_DATASETS.get(task, ()))
    return required


def check_local_tasks(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Validate local benchmark mirrors without contacting the Hub."
    )
    parser.add_argument("--tasks", required=True)
    args = parser.parse_args(argv)
    tasks = [task for task in args.tasks.split(",") if task]
    if not tasks:
        parser.error("--tasks must contain at least one task")

    for repo_id in sorted(datasets_for_tasks(tasks)):
        require_path(local_dataset_path(repo_id), f"dataset mirror for {repo_id}")
    if {"mbpp", "mbpp_instruct"}.intersection(tasks):
        metric_dir = eval_data_root() / "code_eval"
        require_path(metric_dir / "code_eval.py", "local code_eval metric")
        require_path(metric_dir / "execute.py", "local code_eval executor")

    from lm_eval.tasks import TaskManager

    loaded = TaskManager().load(tasks)
    print(
        f"Validated {len(loaded['tasks'])} leaf tasks from local data: "
        f"{', '.join(tasks)}"
    )


def main() -> None:
    install_offline_loaders()
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_local_tasks(sys.argv[2:])
        return

    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
