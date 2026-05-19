"""Runtime patches for HRM-Text Prime-RL experiments.

Python imports this module automatically when this directory is on PYTHONPATH.
The patch is opt-in via HRM_TEXT_PRIME_RL_PATCH_VLLM=1 so importing the reward
environment does not affect unrelated jobs.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import functools
import inspect
import os
import sys
from types import ModuleType
from typing import Any


_TARGET = "vllm.model_executor.models.transformers.base"
_PATCHED = False
_HRM_PREFIXLM_PATCHED: set[str] = set()
_DYNAMIC_MODULE_PATCHED = False


def _patch_hrm_mask_compat() -> None:
    """Bridge HRM-Text PrefixLM masks across nearby Transformers builds."""
    try:
        import transformers.masking_utils as masking_utils
        import transformers.utils.generic as generic
    except Exception as exc:  # pragma: no cover - defensive startup path
        print(f"[hrm-prime-rl] skipped HRM mask compat patch: {exc}", file=sys.stderr)
        return

    if not hasattr(generic, "split_attention_implementation"):

        def split_attention_implementation(attn_implementation: Any) -> tuple[Any, Any]:
            if isinstance(attn_implementation, dict):
                base = (
                    attn_implementation.get("")
                    or attn_implementation.get("base")
                    or attn_implementation.get("default")
                    or next(iter(attn_implementation.values()), None)
                )
                return attn_implementation, base
            return None, attn_implementation

        generic.split_attention_implementation = split_attention_implementation

    if getattr(masking_utils.create_causal_mask, "_hrm_prime_rl_compat", False):
        return

    real_create_causal_mask = masking_utils.create_causal_mask

    def compat_create_causal_mask(
        *args: Any,
        block_sequence_ids: Any = None,
        or_mask_function: Any = None,
        **kwargs: Any,
    ) -> Any:
        if block_sequence_ids is None:
            return real_create_causal_mask(*args, or_mask_function=or_mask_function, **kwargs)

        def prefix_block_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> Any:
            q_block = block_sequence_ids[batch_idx, q_idx]
            kv_block = block_sequence_ids[batch_idx, kv_idx]
            return (q_block >= 0) & (q_block == kv_block)

        if or_mask_function is not None:
            original_or_mask = or_mask_function

            def combined_or_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> Any:
                return prefix_block_mask(batch_idx, head_idx, q_idx, kv_idx) | original_or_mask(
                    batch_idx, head_idx, q_idx, kv_idx
                )

            prefix_or_mask = combined_or_mask
        else:
            prefix_or_mask = prefix_block_mask

        return real_create_causal_mask(*args, or_mask_function=prefix_or_mask, **kwargs)

    compat_create_causal_mask._hrm_prime_rl_compat = True
    masking_utils.create_causal_mask = compat_create_causal_mask
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").endswith("modeling_hrm_text") and hasattr(module, "create_causal_mask"):
            module.create_causal_mask = compat_create_causal_mask
    print("[hrm-prime-rl] patched Transformers PrefixLM mask compatibility", file=sys.stderr)


def _patch_vllm_transformers_base(module: ModuleType) -> None:
    global _PATCHED
    if _PATCHED:
        return

    try:
        import torch
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception as exc:  # pragma: no cover - defensive startup path
        print(f"[hrm-prime-rl] skipped vLLM attention patch: {exc}", file=sys.stderr)
        return

    def hrm_compatible_vllm_flash_attention_forward(
        attn_module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float | None = None,
        attention_instances: dict[int, Any] | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, None]:
        if attention_instances is None:
            raise ValueError("vLLM attention_instances were not provided")

        self_attn = attention_instances[attn_module.layer_idx]
        if scaling is not None:
            self_attn.impl.scale = float(scaling)

        batch_size, num_heads, seq_len, head_dim = query.shape
        query, key, value = (x.transpose(1, 2).reshape(batch_size * seq_len, -1) for x in (query, key, value))
        output = self_attn.forward(query, key, value)

        # vLLM returns [tokens, hidden]. HF attention interfaces return
        # [batch, seq, heads, head_dim], which HRM needs for its sigmoid gate.
        return output.view(batch_size, seq_len, num_heads, head_dim), None

    module.vllm_flash_attention_forward = hrm_compatible_vllm_flash_attention_forward
    ALL_ATTENTION_FUNCTIONS["vllm"] = hrm_compatible_vllm_flash_attention_forward
    _PATCHED = True
    print("[hrm-prime-rl] patched vLLM Transformers attention for HRM-Text", file=sys.stderr)


def _patch_hrm_prefixlm_default(module: ModuleType) -> None:
    if module.__name__ in _HRM_PREFIXLM_PATCHED:
        return
    model_cls = getattr(module, "HrmTextModel", None)
    if model_cls is None or getattr(model_cls, "_prime_rl_prefixlm_patched", False):
        return

    try:
        import torch
    except Exception as exc:  # pragma: no cover - defensive startup path
        print(f"[hrm-prime-rl] skipped HRM PrefixLM patch: {exc}", file=sys.stderr)
        return

    original_forward = model_cls.forward

    @functools.wraps(original_forward)
    def forward_with_prefixlm_prompt_token_types(self: Any, *args: Any, **kwargs: Any) -> Any:
        token_type_ids = kwargs.get("token_type_ids")
        if token_type_ids is None and len(args) >= 5:
            token_type_ids = args[4]

        if (
            token_type_ids is None
            and getattr(self.config, "prefix_lm", False)
            and not self.training
        ):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            inputs_embeds = kwargs.get("inputs_embeds")
            if inputs_embeds is None and len(args) >= 6:
                inputs_embeds = args[5]

            if input_ids is not None:
                token_type_ids = torch.ones_like(input_ids, dtype=torch.long)
            elif inputs_embeds is not None:
                token_type_ids = torch.ones(
                    inputs_embeds.shape[:2],
                    device=inputs_embeds.device,
                    dtype=torch.long,
                )

            if token_type_ids is not None:
                if len(args) >= 5:
                    args = list(args)
                    args[4] = token_type_ids
                    args = tuple(args)
                else:
                    kwargs["token_type_ids"] = token_type_ids

        return original_forward(self, *args, **kwargs)

    forward_with_prefixlm_prompt_token_types.__signature__ = inspect.signature(original_forward)
    model_cls.forward = forward_with_prefixlm_prompt_token_types
    model_cls._prime_rl_prefixlm_patched = True
    _HRM_PREFIXLM_PATCHED.add(module.__name__)
    print("[hrm-prime-rl] patched HRM-Text eval forward to default prompt token_type_ids=1", file=sys.stderr)


def _patch_dynamic_module_loader() -> None:
    """Patch HF dynamic-module loading because it bypasses normal meta_path hooks."""
    global _DYNAMIC_MODULE_PATCHED
    if _DYNAMIC_MODULE_PATCHED:
        return

    try:
        import transformers.dynamic_module_utils as dynamic_module_utils
    except Exception as exc:  # pragma: no cover - defensive startup path
        print(f"[hrm-prime-rl] skipped HF dynamic module patch: {exc}", file=sys.stderr)
        return

    original_get_class = dynamic_module_utils.get_class_from_dynamic_module

    def get_class_from_dynamic_module_with_hrm_patch(*args: Any, **kwargs: Any) -> Any:
        cls = original_get_class(*args, **kwargs)
        module = sys.modules.get(getattr(cls, "__module__", ""))
        if module is not None and getattr(module, "__name__", "").endswith("modeling_hrm_text"):
            _patch_hrm_mask_compat()
            _patch_hrm_prefixlm_default(module)
        return cls

    dynamic_module_utils.get_class_from_dynamic_module = get_class_from_dynamic_module_with_hrm_patch
    _DYNAMIC_MODULE_PATCHED = True
    print("[hrm-prime-rl] patched HF dynamic module loader for HRM-Text", file=sys.stderr)


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader, patch_kind: str) -> None:
        self._wrapped = wrapped
        self._patch_kind = patch_kind

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        create_module = getattr(self._wrapped, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        if self._patch_kind == "vllm":
            _patch_vllm_transformers_base(module)
        elif self._patch_kind == "hrm":
            _patch_hrm_mask_compat()
            _patch_hrm_prefixlm_default(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: list[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname == _TARGET:
            patch_kind = "vllm"
        elif fullname.endswith("modeling_hrm_text"):
            patch_kind = "hrm"
        else:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _PatchLoader(spec.loader, patch_kind)
        return spec


if os.environ.get("HRM_TEXT_PRIME_RL_PATCH_VLLM") == "1":
    _patch_hrm_mask_compat()
    _patch_dynamic_module_loader()
    if _TARGET in sys.modules:
        _patch_vllm_transformers_base(sys.modules[_TARGET])
    for name, module in list(sys.modules.items()):
        if name.endswith("modeling_hrm_text"):
            _patch_hrm_prefixlm_default(module)
    sys.meta_path.insert(0, _PatchFinder())
