"""Uniform CPU replay of generated trajectories and their paired metadata."""

import copy
import random
from collections import deque

import torch
from torch.nn.utils.rnn import pad_sequence


class ReplayBuffer:
    def __init__(self, args):
        self.replay_memory = deque(maxlen=args.capacity)
        self.bs = args.batch_size
        self.pad_id = getattr(args, "pad_token_id", 0)

    def __len__(self):
        return len(self.replay_memory)

    def move_to_memory(self, model_data, no_model_data, gen_data=None):
        for index in range(model_data["input_ids"].shape[0]):
            entry = []
            for batch in (model_data, no_model_data, gen_data or {}):
                entry.append({key: (value[index].detach().cpu().clone()
                                   if isinstance(value, torch.Tensor)
                                   else copy.deepcopy(value[index]))
                              for key, value in batch.items()})
            self.replay_memory.append(entry)

    def sample(self, batch_size=None):
        entries = random.sample(self.replay_memory, batch_size or self.bs)
        result = []
        for column in zip(*entries):
            batch = {}
            for key in column[0]:
                values = [row[key] for row in column]
                if isinstance(values[0], torch.Tensor):
                    padding = -100 if key == "label" else self.pad_id if key == "input_ids" else 0
                    batch[key] = (torch.stack(values) if values[0].ndim == 0 else
                                  pad_sequence(values, batch_first=True, padding_value=padding))
                else:
                    batch[key] = copy.deepcopy(values)
            result.append(batch)
        return tuple(result)

    @staticmethod
    def move_to_device(model_data, no_model_data, gen_data, device):
        for batch in (model_data, no_model_data, gen_data):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
        return model_data, no_model_data, gen_data
