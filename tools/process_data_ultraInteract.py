import multiprocessing
import os
import time
import json
import sys
import random
from pathlib import Path
from data_utils.records import get_raw_prompt, get_response

random.seed(42)


def reference_response(record):
    """Use the same first reference for tokenization and JSONL metadata."""
    response = get_response(record)
    if isinstance(response, list):
        response = response[0] if response else None
    return response if isinstance(response, str) and response.strip() else None


def read_raw_data(path):
    records = []
    skipped_lines = []
    with open(path, "r", encoding="utf-8") as source:
        for line_number, text in enumerate(source, 1):
            if not text.strip():
                continue
            record = json.loads(text)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object")
            if reference_response(record) is None:
                skipped_lines.append(line_number)
                continue
            records.append(record)
    return records, skipped_lines


# 1. Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.
class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        from transformers import AutoTokenizer

        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

    def encode(self, line):
        raw_prompt = get_raw_prompt(line, Encoder.tokenizer)
        response = reference_response(line)
        if response is None:
            raise ValueError("Preprocessing requires a nonempty reference response")

        messages = []
        if line.get("system_prompt"):
            messages.append({"role": "system", "content": line["system_prompt"]})
        messages.append({"role": "user", "content": raw_prompt})
        
        prompt_str = Encoder.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        # Insert teacher-only context inside the user turn, before the assistant header.
        privileged_prompt = Encoder.tokenizer.apply_chat_template(
            messages[:-1] + [{"role": "user", "content": raw_prompt + "{privileged_context}"}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize the two causal segments independently. Tokenizing their text
        # concatenation can create a BPE token across the boundary, making the
        # response IDs disagree with JSONL loading and step-span alignment.
        prompt_tokens = Encoder.tokenizer.encode(prompt_str, add_special_tokens=False)
        response_tokens = Encoder.tokenizer.encode(response, add_special_tokens=False) + [
            Encoder.tokenizer.eos_token_id]

        # Match data_utils.lm_datasets._pack_example: retain the suffix so chat
        # generation headers immediately before the response are not discarded.
        prompt_tokens = prompt_tokens[-self.args.max_prompt_length:]
            
        bytes_processed = len(prompt_str.encode('utf-8')) + len(response.encode('utf-8'))

        return line, prompt_str, privileged_prompt, prompt_tokens, response_tokens, bytes_processed


def processed_record(line, prompt, privileged_prompt):
    # Raw question validity was already checked in the encoding worker.
    record = dict(instruction=get_raw_prompt(line, None), prompt=prompt,
                  privileged_prompt=privileged_prompt, output=reference_response(line))
    if "context" in line:
        if not isinstance(line["context"], str) or not line["context"].strip():
            raise ValueError("Prepared raw rows must contain nonempty context")
        record["context"] = line["context"]
    if line.get("system_prompt"):
        record["system_prompt"] = line["system_prompt"]
    return record


def resolve_processed_data_dir(processed_data_dir, model_path, base_path=None):
    model_path = Path(model_path)
    if model_path.is_absolute():
        project_root = Path(base_path or Path.cwd()).resolve()
        try:
            model_subdir = model_path.resolve().relative_to(project_root)
        except ValueError:
            model_subdir = Path("models") / model_path.name
    else:
        model_subdir = model_path
    if ".." in model_subdir.parts:
        model_subdir = Path("models") / model_path.name
    return str(Path(processed_data_dir) / model_subdir)


def main():
    import torch
    import numpy as np
    from data_utils.indexed_dataset import make_builder
    from arguments import get_args

    print("OK")
    args = get_args()

    args.processed_data_dir = resolve_processed_data_dir(
        args.processed_data_dir,
        args.model_path,
        args.base_path,
    )

    os.makedirs(args.processed_data_dir, exist_ok=True)

    print(f"Reading data from: {args.data_dir}")
    raw_data, skipped_lines = read_raw_data(args.data_dir)
    if skipped_lines:
        preview = ", ".join(map(str, skipped_lines[:10]))
        suffix = ", ..." if len(skipped_lines) > 10 else ""
        print(f"Skipped {len(skipped_lines)} rows with empty/missing reference responses "
              f"(JSONL lines: {preview}{suffix}).", file=sys.stderr)
    print(f"Total data instances: {len(raw_data)}")
    if not raw_data:
        raise ValueError("No rows with nonempty reference responses remain")
    
    if args.dev_num > 0:
        if args.dev_num >= len(raw_data):
            raise ValueError("--dev-num must leave at least one training example")
        valid_indices = set(random.Random(args.seed).sample(range(len(raw_data)), args.dev_num))
        valid_data = [row for index, row in enumerate(raw_data) if index in valid_indices]
        train_data = [row for index, row in enumerate(raw_data) if index not in valid_indices]
        all_data = {
            "valid": valid_data,
            "train": train_data
        }
    else:
        all_data = {
            "train": raw_data
        }

    for split in all_data:

        # encoder use the tokenizer to encode data
        encoder = Encoder(args)

        # 2. Mapping all datas with Encoder, with the help of multiprocessing
        pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
        encoded_docs = pool.imap_unordered(encoder.encode, all_data[split], chunksize=50)
        proc_start = time.time()
        total_bytes_processed = 0

        bin_file = os.path.join(args.processed_data_dir, f"{split}_{0}.bin")
        idx_file = os.path.join(args.processed_data_dir, f"{split}_{0}.idx")

        if args.model_type != "qwen":
            binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)
        else:
            binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint32)

        # put tokenized data into binary_builder
        inst_num = 0
        print("#" * 10, split, "#" * 10)

        prompt_lens = []
        response_lens = []

        json_file = open(os.path.join(args.processed_data_dir, f"{split}.jsonl"), "w")

        for lid, (line, prompt_str, privileged_prompt, prompt, response, bytes_processed) in enumerate(encoded_docs):
            total_bytes_processed += bytes_processed
            if prompt is None:
                continue

            if args.only_prompt:
                if len(prompt) < args.max_length:
                    binary_builder.add_item(torch.IntTensor(prompt))
                else:
                    continue
            else:
                binary_builder.add_item(torch.IntTensor(prompt + [-1] + response))

            # Carry context with its own row through splitting and unordered workers.
            # The raw-source fingerprint stays in the original prepared JSONL;
            # training reads context directly from these processed records.
            json_file.write(json.dumps(processed_record(
                line, prompt_str, privileged_prompt)) + "\n")

            prompt_lens.append(len(prompt))
            response_lens.append(len(response))

            inst_num += 1
            if lid % 1000 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {lid} documents. {inst_num} instances.",
                      f"({lid / elapsed:.2f} docs/s, {mbs:.4f} MB/s).",
                      file=sys.stderr)

        # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
        binary_builder.finalize(idx_file)

        # close multiproceessing mapping
        pool.close()
        json_file.close()

        print("Data num", len(prompt_lens))
        print("Prompt lengths.", "Mean:", np.mean(prompt_lens), "Max:", np.max(prompt_lens), "Min:",
              np.min(prompt_lens))
        print("Response", "Mean:", np.mean(response_lens), "Max:", np.max(response_lens), "Min:", np.min(response_lens))


if __name__ == '__main__':
    main()
