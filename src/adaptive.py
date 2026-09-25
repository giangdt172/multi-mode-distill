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
    ROUTING_ORDER = ("on_policy", "self_distill", "off_policy")
    MODE_SETS = {
        "all": MODES,
        "on_self": ("self_distill", "on_policy"),
        "off_self": ("off_policy", "self_distill"),
    }
    MONITORED_MODE_SETS = {
        "all": ("self_distill", "on_policy"),
        "on_self": ("on_policy",),
        "off_self": ("self_distill",),
    }

    def __init__(self, config=None, seed=42, mode_set="all"):
        self.config = config or AdaptiveConfig()
        if mode_set not in self.MODE_SETS:
            raise ValueError(
                f"Unknown adaptive mode set {mode_set!r}; "
                f"choose one of {', '.join(self.MODE_SETS)}")
        self.mode_set = mode_set
        self.enabled_modes = self.MODE_SETS[mode_set]
        self.monitored_modes = self.MONITORED_MODE_SETS[mode_set]
        self.rng = random.Random(seed)
        self.rho_self = self.config.rho_self_init
        self.rho_on = self.config.rho_on_init
        self.ref_self_loss = None
        self.ref_on_loss = None
        self.counts = dict.fromkeys(self.MODES, 0)

    @property
    def rho_off(self):
        return self.mode_probabilities["off_policy"]

    @property
    def mode_probabilities(self):
        """Return probabilities, with one adaptive degree of freedom per pair."""
        if self.mode_set == "on_self":
            return {
                "off_policy": 0.0,
                "self_distill": 1.0 - self.rho_on,
                "on_policy": self.rho_on,
            }
        if self.mode_set == "off_self":
            return {
                "off_policy": 1.0 - self.rho_self,
                "self_distill": self.rho_self,
                "on_policy": 0.0,
            }
        return {
            "off_policy": 1.0 - self.rho_self - self.rho_on,
            "self_distill": self.rho_self,
            "on_policy": self.rho_on,
        }

    def sample_mode(self, device=None):
        """Rank zero draws once; every rank receives the same categorical mode."""
        distributed = dist.is_available() and dist.is_initialized()
        mode_id = 0
        if not distributed or dist.get_rank() == 0:
            u = self.rng.random()
            cumulative = 0.0
            probabilities = self.mode_probabilities
            selected_mode = self.ROUTING_ORDER[-1]
            for mode in self.ROUTING_ORDER:
                if probabilities[mode] > 0:
                    selected_mode = mode
                cumulative += probabilities[mode]
                if u < cumulative:
                    selected_mode = mode
                    break
            mode_id = self.MODES.index(selected_mode)
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

    def on_evaluation(self, self_loss=None, on_loss=None):
        losses = {"self_distill": self_loss, "on_policy": on_loss}
        monitored_modes = self.monitored_modes
        for mode in monitored_modes:
            loss = losses[mode]
            if loss is None:
                raise ValueError(f"Missing adaptive evaluation loss for {mode}")
            if not math.isfinite(loss):
                raise FloatingPointError("Non-finite adaptive evaluation loss")
        self_deterioration = on_deterioration = None
        self_updated = on_updated = False
        c = self.config
        if "self_distill" in monitored_modes:
            if self.ref_self_loss is None:
                self.ref_self_loss = self_loss
            else:
                self_deterioration = (self_loss - self.ref_self_loss) / (abs(self.ref_self_loss) + c.eps)
                if self_deterioration > c.deterioration_threshold:
                    new_rho = min(self.rho_self + c.rho_self_increment, c.rho_self_max)
                    self_updated = new_rho > self.rho_self
                    self.rho_self = new_rho
                    self.ref_self_loss = self_loss
                else:
                    self.ref_self_loss = min(self.ref_self_loss, self_loss)
        if "on_policy" in monitored_modes:
            if self.ref_on_loss is None:
                self.ref_on_loss = on_loss
            else:
                on_deterioration = (on_loss - self.ref_on_loss) / (abs(self.ref_on_loss) + c.eps)
                if on_deterioration > c.deterioration_threshold:
                    new_rho = min(self.rho_on + c.rho_on_increment, c.rho_on_max)
                    on_updated = new_rho > self.rho_on
                    self.rho_on = new_rho
                    self.ref_on_loss = on_loss
                else:
                    self.ref_on_loss = min(self.ref_on_loss, on_loss)
        probabilities = self.mode_probabilities
        return {
            "scheduler/mode_set": self.mode_set,
            "scheduler/enabled_modes": list(self.enabled_modes),
            "scheduler/monitored_modes": list(self.monitored_modes),
            "scheduler/rho_off": probabilities["off_policy"],
            "scheduler/rho_self": probabilities["self_distill"],
            "scheduler/rho_on": probabilities["on_policy"],
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
