"""One categorical exposure decision per optimizer step, guided by dev discrepancies."""

from dataclasses import dataclass, fields
import math
import random

import torch
import torch.distributed as dist


DEPRECATED_ADAPTIVE_ARGUMENTS = (
    "rho_min", "rho_max", "rho_increment", "progress_signal", "scheduler_metric",
    "loss_improvement_threshold", "metric_improvement_threshold", "eval_ema_beta",
    "min_evals_between_rho_updates", "off_ema_beta", "off_min_absorption",
    "off_plateau_threshold", "off_transition_patience",
)


@dataclass(frozen=True)
class AdaptiveConfig:
    rho_self_init: float = 0.05
    rho_on_init: float = 0.05
    rho_self_max: float = 0.25
    rho_on_max: float = 0.25
    rho_self_increment: float = 0.025
    rho_on_increment: float = 0.025
    deterioration_threshold: float = 0.05
    eps: float = 1e-8

    @classmethod
    def from_args(cls, args):
        values = {field.name: getattr(args, field.name, field.default) for field in fields(cls)}
        values["deterioration_threshold"] = getattr(
            args, "adaptive_deterioration_threshold", cls.deterioration_threshold)
        values["eps"] = getattr(args, "adaptive_eps", cls.eps)
        return cls(**values)

    def __post_init__(self):
        for field in fields(self):
            if not math.isfinite(getattr(self, field.name)):
                raise ValueError(f"--{field.name.replace('_', '-')} must be finite")
        if not (0 <= self.rho_self_init <= self.rho_self_max
                and 0 <= self.rho_on_init <= self.rho_on_max
                and self.rho_self_max + self.rho_on_max <= 1):
            raise ValueError("Adaptive rho initial values must fit their caps; caps must sum to at most 1")
        if self.rho_self_increment <= 0 or self.rho_on_increment <= 0:
            raise ValueError("Adaptive rho increments must be positive")
        if self.deterioration_threshold < 0 or self.eps <= 0:
            raise ValueError("Deterioration threshold must be nonnegative and adaptive eps positive")


class AdaptiveScheduler:
    MODES = ("off_policy", "self_distill", "on_policy")

    def __init__(self, config=None, seed=42):
        self.config = config or AdaptiveConfig()
        self.rng = random.Random(seed)
        self.rho_self = self.config.rho_self_init
        self.rho_on = self.config.rho_on_init
        self.ref_self_loss = None
        self.ref_on_loss = None
        self.counts = dict.fromkeys(self.MODES, 0)

    @property
    def rho_off(self):
        return 1.0 - self.rho_self - self.rho_on

    def sample_mode(self, device=None):
        """Rank zero draws once; every rank receives the same categorical mode."""
        distributed = dist.is_available() and dist.is_initialized()
        mode_id = 0
        if not distributed or dist.get_rank() == 0:
            u = self.rng.random()
            mode_id = 2 if u < self.rho_on else 1 if u < self.rho_on + self.rho_self else 0
        if distributed:
            routing_device = (device if device is not None else torch.cuda.current_device()) \
                if dist.get_backend() == "nccl" else "cpu"
            decision = torch.tensor(mode_id, dtype=torch.long, device=routing_device)
            dist.broadcast(decision, src=0)
            mode_id = int(decision.item())
        return self.MODES[mode_id]

    def record_step(self, mode):
        """Count completed optimizer updates, not micro-batches or unfinished windows."""
        if mode not in self.counts:
            raise ValueError(f"Unknown distillation mode: {mode}")
        self.counts[mode] += 1

    def on_evaluation(self, self_loss, on_loss):
        if not math.isfinite(self_loss) or not math.isfinite(on_loss):
            raise FloatingPointError("Non-finite adaptive evaluation loss")
        self_deterioration = on_deterioration = None
        self_updated = on_updated = False
        if self.ref_self_loss is None:
            self.ref_self_loss = self_loss
            self.ref_on_loss = on_loss
        else:
            c = self.config
            self_deterioration = (self_loss - self.ref_self_loss) / (abs(self.ref_self_loss) + c.eps)
            on_deterioration = (on_loss - self.ref_on_loss) / (abs(self.ref_on_loss) + c.eps)
            if self_deterioration > c.deterioration_threshold:
                new_rho = min(self.rho_self + c.rho_self_increment, c.rho_self_max)
                self_updated = new_rho > self.rho_self
                self.rho_self = new_rho
                self.ref_self_loss = self_loss
            else:
                self.ref_self_loss = min(self.ref_self_loss, self_loss)
            if on_deterioration > c.deterioration_threshold:
                new_rho = min(self.rho_on + c.rho_on_increment, c.rho_on_max)
                on_updated = new_rho > self.rho_on
                self.rho_on = new_rho
                self.ref_on_loss = on_loss
            else:
                self.ref_on_loss = min(self.ref_on_loss, on_loss)
        return {
            "scheduler/rho_off": self.rho_off,
            "scheduler/rho_self": self.rho_self,
            "scheduler/rho_on": self.rho_on,
            "scheduler/eval_self_loss": self_loss,
            "scheduler/eval_on_loss": on_loss,
            "scheduler/ref_self_loss": self.ref_self_loss,
            "scheduler/ref_on_loss": self.ref_on_loss,
            "scheduler/self_deterioration": self_deterioration,
            "scheduler/on_deterioration": on_deterioration,
            "scheduler/self_updated": self_updated,
            "scheduler/on_updated": on_updated,
            "scheduler/off_steps": self.counts["off_policy"],
            "scheduler/self_distill_steps": self.counts["self_distill"],
            "scheduler/on_policy_steps": self.counts["on_policy"],
        }


class OptimizerStepModeRouter:
    """Hold one sampled mode until the optimizer finishes its accumulation window."""

    def __init__(self, scheduler=None, fixed_mode=None):
        self.scheduler = scheduler
        self.fixed_mode = fixed_mode
        self.current_mode = None

    def for_microbatch(self, device=None):
        if self.current_mode is None:
            self.current_mode = (self.scheduler.sample_mode(device) if self.scheduler
                                 else self.fixed_mode)
        return self.current_mode

    def on_optimizer_step(self):
        if self.current_mode is None:
            raise RuntimeError("No mode was selected for this optimizer step")
        if self.scheduler:
            self.scheduler.record_step(self.current_mode)
        self.current_mode = None
