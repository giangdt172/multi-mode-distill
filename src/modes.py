"""Validation and response alignment helpers for finetune_v2."""

import math

import torch

def geometry_enabled_for_mode(args, mode):
    return ((args.geometry or getattr(args, "cka", False))
            and mode in ("off_policy", "self_distill", "on_policy")) or (
        args.off_policy_geometry and mode == "off_policy")


def menger_enabled_for_mode(args, mode):
    """Menger distillation is defined for the three teacher/reference modes."""
    return (getattr(args, "menger_weight", 0.0) > 0
            and mode in ("off_policy", "self_distill", "on_policy"))


def validate_mode_args(args):
    from .adaptive import DEPRECATED_ADAPTIVE_ARGUMENTS
    old_flags = [name.replace("_", "-") for name in DEPRECATED_ADAPTIVE_ARGUMENTS
                 if hasattr(args, name)]
    if old_flags:
        raise ValueError("Old adaptive scheduler arguments were removed: " +
                         ", ".join("--" + name for name in old_flags))
    adaptive = getattr(args, "adaptive_on_policy", False)
    adaptive_mode_set = getattr(args, "adaptive_mode_set", "all")
    if adaptive_mode_set != "all" and not adaptive:
        raise ValueError("--adaptive-mode-set requires --dual-adaptive-exposure")
    if adaptive:
        from .adaptive import AdaptiveConfig
        AdaptiveConfig.from_args(args)
        if args.student_gen or args.distill_mode not in (None, "off_policy") or args.type not in (None, "kd"):
            raise ValueError("--adaptive-on-policy requires --type kd and no fixed ON/self mode or --student-gen")
        if not args.teacher_model_path:
            raise ValueError("--adaptive-on-policy requires --teacher-model-path")
        if args.do_train and (not args.eval_interval or args.eval_interval < -1):
            raise ValueError("--adaptive-on-policy requires positive --eval-interval (or -1 for each epoch)")
        if not getattr(args, "do_train", False):
            raise ValueError("Adaptive exposure requires --do-train")
    legacy_type = args.type or "kd"
    if "adaptive" in legacy_type or "mixed" in legacy_type:
        raise ValueError("Use --distill-mode; "
                         "adaptive/mixed routing is not supported by finetune_v2")
    if legacy_type == "rvd":
        raise ValueError("Use finetune.py for --type rvd; finetune_v2 uses --type kd")
    if args.student_gen:
        if args.distill_mode not in (None, "on_policy"):
            raise ValueError("--student-gen conflicts with --distill-mode")
        args.distill_mode = "on_policy"
    args.distill_mode = args.distill_mode or (
        legacy_type if legacy_type in ("on_policy", "self_distill", "opsd") else "off_policy"
    )
    if args.kd_loss is None:
        args.kd_loss = next((name for name in ("sfkl", "srkl", "jsd", "tvd", "fkl", "rkl")
                             if name in legacy_type), None)
        if args.kd_loss is None:
            if legacy_type not in ("kd", "lm", "off_policy", "on_policy", "self_distill", "opsd"):
                raise ValueError("Specify --kd-loss for this legacy --type")
            args.kd_loss = "fkl"
    if args.kd_ratio is None:
        args.kd_ratio = 1.0
    if not math.isfinite(args.kd_ratio) or not 0 <= args.kd_ratio <= 1:
        raise ValueError("--kd-ratio must be in [0, 1]")
    if not math.isfinite(args.skew_alpha) or not 0 <= args.skew_alpha <= 1:
        raise ValueError("--skew-alpha must be in [0, 1]")
    args.distill_top_k = getattr(args, "distill_top_k", 32)
    if args.distill_top_k < 2:
        raise ValueError("--distill-top-k must be at least 2")
    args.distill_temperature = getattr(args, "distill_temperature", 1.0)
    if not math.isfinite(args.distill_temperature) or args.distill_temperature <= 0:
        raise ValueError("--distill-temperature must be finite and positive")
    args.cka = getattr(args, "cka", False)
    args.cka_weight = getattr(args, "cka_weight", 1.0)
    args.menger_weight = getattr(args, "menger_weight", 0.0)
    args.menger_eps = getattr(args, "menger_eps", 1.0e-6)
    uses_generation = args.distill_mode in ("on_policy", "opsd")
    if args.cka and (args.geometry or args.off_policy_geometry):
        raise ValueError("--cka conflicts with --geometry and --off-policy-geometry")
    if args.off_policy_geometry and args.distill_mode != "off_policy" and not args.geometry:
        raise ValueError("--off-policy-geometry applies only to off_policy")
    if args.geometry and args.distill_mode not in ("off_policy", "self_distill", "on_policy") and not adaptive:
        raise ValueError("--geometry applies to off_policy, self_distill, and on_policy only")
    if args.cka and args.distill_mode not in ("off_policy", "self_distill", "on_policy") and not adaptive:
        raise ValueError("--cka applies to off_policy, self_distill, and on_policy only")
    if any(not math.isfinite(w) or w < 0 for w in (
            args.mag_weight, args.gram_weight, args.cka_weight, args.menger_weight)):
        raise ValueError("Geometry/CKA/Menger weights must be finite and nonnegative")
    if not math.isfinite(args.eps) or args.eps <= 0:
        raise ValueError("--eps must be finite and positive")
    if not math.isfinite(args.menger_eps) or args.menger_eps <= 0:
        raise ValueError("--menger-eps must be finite and positive")
    if args.do_train and not args.teacher_model_path and legacy_type != "lm" and args.distill_mode not in ("self_distill", "opsd"):
        raise ValueError("Distillation requires --teacher-model-path")
    if legacy_type == "lm" and (uses_generation or args.distill_mode == "self_distill" or args.disable_lm_loss):
        raise ValueError("--type lm requires canonical LM supervision")
    if not 0 < args.max_prompt_length < args.max_length:
        raise ValueError("Require 0 < --max-prompt-length < --max-length")
    if (args.distill_mode in ("self_distill", "opsd") or adaptive) and not 0 < args.t_max_prompt_length < args.t_max_length:
        raise ValueError("Require 0 < --t-max-prompt-length < --t-max-length")
    if (args.distill_mode in ("self_distill", "opsd") or adaptive) and args.t_max_length < args.max_length:
        raise ValueError("Self-distillation reference length must fit the full student response")
    if not 0 <= args.self_distill_context_drop_ratio <= 1 or not math.isfinite(args.self_distill_context_drop_ratio):
        raise ValueError("--self-distill-context-drop-ratio must be finite and in [0, 1]")
    args.self_distill_context_fixed_drop_ratio = getattr(
        args, "self_distill_context_fixed_drop_ratio", None)
    fixed_drop_ratio = args.self_distill_context_fixed_drop_ratio
    if (fixed_drop_ratio is not None
            and (not math.isfinite(fixed_drop_ratio)
                 or not 0 <= fixed_drop_ratio <= 1)):
        raise ValueError(
            "--self-distill-context-fixed-drop-ratio must be finite and in [0, 1]")
    if args.self_distill_context_max_tokens < 0:
        raise ValueError("--self-distill-context-max-tokens must be nonnegative")
    if "{context}" not in args.self_distill_context_template:
        raise ValueError("--self-distill-context-template must include {context}")
    try:
        args.self_distill_context_template.format(context="")
    except (KeyError, ValueError) as error:
        raise ValueError("Invalid --self-distill-context-template") from error
    if args.distill_mode == "opsd":
        if args.peft != "lora":
            raise ValueError("OPSD fixed teacher requires --peft lora")
        if args.teacher_model_path:
            raise ValueError("OPSD fixed teacher uses the student's base model; omit --teacher-model-path")
        if args.geometry or args.off_policy_geometry or args.cka:
            raise ValueError("OPSD ablation uses token divergence only; disable geometry and CKA")
        if not args.disable_lm_loss:
            raise ValueError("OPSD ablation requires --disable-lm-loss")
        if args.kd_loss != "fkl":
            raise ValueError("OPSD ablation uses forward KL; set --kd-loss fkl")
        if args.distill_temperature != 1.0:
            raise ValueError("OPSD ablation uses temperature 1.0")
        if not math.isfinite(args.opsd_token_clip) or args.opsd_token_clip < 0:
            raise ValueError("--opsd-token-clip must be finite and nonnegative")
        if "{context}" not in args.opsd_context_template:
            raise ValueError("--opsd-context-template must include {context}")
        try:
            args.opsd_context_template.format(context="solution")
        except (KeyError, ValueError) as error:
            raise ValueError("Invalid --opsd-context-template") from error
    if args.do_train and (args.batch_size < 1 or args.gradient_accumulation_steps < 1 or args.log_interval < 1):
        raise ValueError("Batch size, gradient accumulation and log interval must be positive")


def require_shared_vocabulary(student_tokenizer, teacher_tokenizer):
    if (student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab()
            or student_tokenizer.eos_token_id != teacher_tokenizer.eos_token_id):
        raise ValueError("Token-level KD requires matching vocabulary/token IDs and EOS; "
                         "different tokenizers need an explicit vocabulary mapping")


def align_response_logits(student_logits, student_labels, teacher_logits, teacher_labels):
    """[B,Ls,V], [B,Ls], [B,Lt,V], [B,Lt] -> [N,V], [N,V], [N].

    Labels already contain next-token targets. Compare prediction positions by
    response index, including EOS. Reject mismatches rather than silently crop.
    """
    if student_logits.ndim != 3 or teacher_logits.ndim != 3:
        raise ValueError("Expected student/teacher logits [B, L, V]")
    if student_logits.shape[:2] != student_labels.shape or teacher_logits.shape[:2] != teacher_labels.shape:
        raise ValueError("Logit sequence dimensions must match labels")
    if (student_logits.shape[0] != teacher_logits.shape[0]):
        raise ValueError("Batch size must match")
    sm, tm = student_labels != -100, teacher_labels != -100
    if not torch.equal(sm.sum(-1), tm.sum(-1)):
        raise ValueError("Student/teacher response lengths differ")
    if not torch.equal(student_labels[sm], teacher_labels[tm]):
        raise ValueError("Student/teacher response token IDs differ")
    return student_logits[sm], teacher_logits[tm], student_labels[sm]
