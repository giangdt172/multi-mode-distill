"""Response field aliases shared by training, evaluation and context preparation."""

RESPONSE_FIELDS = ("output", "response", "generated_text")


def get_raw_prompt(record, tokenizer):
    """Read raw questions while refusing to nest an already rendered chat."""
    field = next((key for key in ("user_prompt", "instruction", "prompt")
                  if isinstance(record.get(key), str) and record[key].strip()), None)
    if field is None:
        raise ValueError("Provide a nonempty user_prompt/instruction/prompt")
    prompt = record[field]
    if field == "prompt" and getattr(tokenizer, "chat_template", None):
        if any(marker and marker in prompt for marker in getattr(tokenizer, "all_special_tokens", [])):
            raise ValueError("Rendered chat prompt requires raw 'user_prompt' or 'instruction'; "
                             "do not wrap chat headers twice")
    return prompt


def get_response(record):
    """Preserve legacy field precedence without modifying the source record."""
    for field in RESPONSE_FIELDS:
        if field in record:
            return record[field]
    return None


def get_references(record):
    response = get_response(record)
    answers = response if isinstance(response, list) else [response]
    if not answers or any(not isinstance(answer, str) or not answer.strip() for answer in answers):
        raise ValueError("'output', 'response' or 'generated_text' must contain nonempty text references")
    return answers
