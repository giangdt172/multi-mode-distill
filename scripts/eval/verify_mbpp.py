#!/usr/bin/env python3
"""Sanitize and syntax-check MBPP predictions before code execution.

Gemma can occasionally emit the SentencePiece whitespace marker U+2581
(``▁``) literally.  Python does not accept that character as indentation, so
replace it with an ASCII space before passing candidates to ``code_eval``.

This module deliberately does not remove ASCII underscores: names such as
``__init__`` and ``__name__`` are valid Python and must be preserved.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Iterator


SENTENCEPIECE_WHITESPACE = "\u2581"


def clean_mbpp_code(code: str) -> tuple[str, int]:
    """Return code with literal SentencePiece whitespace markers normalized."""
    replacements = code.count(SENTENCEPIECE_WHITESPACE)
    return code.replace(SENTENCEPIECE_WHITESPACE, " "), replacements


def clean_predictions(value: Any) -> tuple[Any, int]:
    """Recursively clean the nested prediction structure used by code_eval."""
    if isinstance(value, str):
        return clean_mbpp_code(value)
    if isinstance(value, list):
        cleaned: list[Any] = []
        replacement_count = 0
        for item in value:
            clean_item, item_count = clean_predictions(item)
            cleaned.append(clean_item)
            replacement_count += item_count
        return cleaned, replacement_count
    if isinstance(value, tuple):
        cleaned, replacement_count = clean_predictions(list(value))
        return tuple(cleaned), replacement_count
    return value, 0


def iter_prediction_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_prediction_strings(item)


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return False
    return True


class SanitizedCodeEvalMetric:
    def __init__(self, metric: Any) -> None:
        self._metric = metric

    def __getattr__(self, name: str) -> Any:
        return getattr(self._metric, name)

    def compute(self, *args: Any, **kwargs: Any) -> Any:
        predictions = kwargs.get("predictions")
        if predictions is None:
            return self._metric.compute(*args, **kwargs)

        cleaned, replacement_count = clean_predictions(predictions)
        kwargs["predictions"] = cleaned
        candidates = list(iter_prediction_strings(cleaned))
        invalid_count = sum(not is_valid_python(code) for code in candidates)
        print(
            "MBPP verify: "
            f"candidates={len(candidates)}, "
            f"sentencepiece_markers_replaced={replacement_count}, "
            f"syntax_invalid_after_clean={invalid_count}",
            file=sys.stderr,
        )
        return self._metric.compute(*args, **kwargs)


def wrap_code_eval_metric(metric: Any) -> SanitizedCodeEvalMetric:
    return SanitizedCodeEvalMetric(metric)


def verify_jsonl(path: Path, output: Path | None = None) -> int:
    records = 0
    candidates = 0
    replacements = 0
    valid_before = 0
    valid_after = 0
    cleaned_lines: list[str] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error

            records += 1
            raw_predictions = record.get("filtered_resps", record.get("resps", []))
            before = list(iter_prediction_strings(raw_predictions))
            valid_before += sum(is_valid_python(code) for code in before)
            candidates += len(before)

            for field in ("resps", "filtered_resps"):
                if field in record:
                    record[field], field_replacements = clean_predictions(record[field])
                    replacements += field_replacements

            after_predictions = record.get("filtered_resps", record.get("resps", []))
            valid_after += sum(
                is_valid_python(code) for code in iter_prediction_strings(after_predictions)
            )
            if output is not None:
                cleaned_lines.append(json.dumps(record, ensure_ascii=False))

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(cleaned_lines) + "\n", encoding="utf-8")

    print(f"records={records}")
    print(f"candidates={candidates}")
    print(f"sentencepiece_markers_replaced={replacements}")
    print(f"syntax_valid_before={valid_before}/{candidates}")
    print(f"syntax_valid_after={valid_after}/{candidates}")
    if output is not None:
        print(f"cleaned_output={output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean U+2581 markers and syntax-check MBPP JSONL predictions."
    )
    parser.add_argument("input", type=Path, help="lm-eval MBPP sample JSONL")
    parser.add_argument(
        "--output",
        type=Path,
        help="optionally write a cleaned JSONL copy; the input is never overwritten",
    )
    args = parser.parse_args()
    return verify_jsonl(args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
