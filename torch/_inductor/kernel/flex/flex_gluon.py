"""Gluon (architecture-specific) forward template for FlexAttention.

BACKEND="GLUON" uses a hand-written Gluon template when the device's arch has
one and the inputs are supported, and otherwise falls back to the Triton
template. A Gluon forward writes the same out/LSE/MAX as the Triton forward, so
the Triton backward consumes it unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch._inductor.virtualized import V
from torch._logging import warning_once
from torch.utils._triton import has_gluon

from ... import config
from ...codegen.gluon_template import GluonTemplate
from ...select_algorithm import (
    autotune_select_algorithm,
    NoValidChoicesError,
    SymbolicGridFn,
)
from ...utils import get_num_sms
from .common import load_flex_template
from .flex_attention import log


if TYPE_CHECKING:
    from collections.abc import Callable

    from ...ir import TensorBox


@dataclasses.dataclass(frozen=True)
class GluonFlexConfig:
    block_m: int
    block_n: int
    num_warps: int
    # Persistent programs launched per CU.
    programs_per_cu: int


@SymbolicGridFn
def persistent_grid(batch_size, q_heads, num_queries, d_model, meta, *, cdiv):
    return (meta["NUM_PROGRAMS"], 1, 1)


@dataclasses.dataclass(frozen=True)
class GluonFlexTemplateSpec:
    template: GluonTemplate
    head_dims: tuple[int, ...]
    dtypes: tuple[torch.dtype, ...]
    configs: tuple[GluonFlexConfig, ...]
    # XCDs that workgroups are dispatched round-robin across.
    num_xcds: int
    # Template-specific constraints beyond head dim and dtype; returns a reason or None.
    check: Callable[[Any, Any, Any, dict[str, Any]], str | None] | None = None


_GFX950_SPEC = GluonFlexTemplateSpec(
    template=GluonTemplate(
        name="gluon_flex_attention_gfx950",
        grid=persistent_grid,
        source=load_flex_template("gluon_flex_attention_gfx950"),
        always_freeze_layout=True,
    ),
    head_dims=(64, 128),
    dtypes=(torch.float16, torch.bfloat16),
    # The [32, 32, 16] MFMA tiles BLOCK_M over num_warps warps of 32 rows each.
    # BLOCK_N = 32 wins causal D = 128 and 64 wins the rest. 4 programs per CU
    # was best or within 2% on every MI350X shape benchmarked; 2 and 3 were not
    # worth autotuning.
    configs=(GluonFlexConfig(128, 64, 4, 4), GluonFlexConfig(128, 32, 4, 4)),
    num_xcds=8,
)

# Keyed by base arch, e.g. "gfx950" for "gfx950:sramecc+:xnack-".
GLUON_FLEX_FWD_TEMPLATES: dict[str, GluonFlexTemplateSpec] = {
    "gfx950": _GFX950_SPEC,
}

_INT32_MAX = 2**31 - 1
# Layouts are derived from these, so a user-forced value must match a config.
_FORCEABLE_OPTIONS = ("BLOCK_M", "BLOCK_N", "num_warps")


def _matches_forced_options(conf: GluonFlexConfig, options: dict[str, Any]) -> bool:
    values = (conf.block_m, conf.block_n, conf.num_warps)
    return all(options.get(k, v) == v for k, v in zip(_FORCEABLE_OPTIONS, values))


def _offsets_fit_int32(node) -> bool:
    # Buffer ops address one (batch, head) slice with 32-bit offsets.
    _, _, seq_len, head_dim = node.get_size()
    _, _, stride_seq, stride_d = node.get_stride()
    extent = (seq_len - 1) * stride_seq + (head_dim - 1) * stride_d
    return V.graph.sizevars.guard_or_false(sympy.Le(extent, _INT32_MAX))


def _num_programs(spec: GluonFlexTemplateSpec, conf: GluonFlexConfig) -> int:
    return conf.programs_per_cu * get_num_sms() // spec.num_xcds * spec.num_xcds


def _get_gluon_spec(
    query, key, value, kernel_options: dict[str, Any]
) -> tuple[GluonFlexTemplateSpec | None, str]:
    if not torch.version.hip or not has_gluon():
        return None, "it requires ROCm and a Triton with Gluon"
    device = query.get_device()
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    spec = GLUON_FLEX_FWD_TEMPLATES.get(arch)
    if spec is None:
        return None, f"there is no Gluon template for {arch}"

    dtype = query.get_dtype()
    if dtype not in spec.dtypes or any(n.get_dtype() != dtype for n in (key, value)):
        return None, f"Q, K and V must share one dtype in {spec.dtypes}"
    qk_head_dim = kernel_options["QK_HEAD_DIM"]
    v_head_dim = kernel_options["V_HEAD_DIM"]
    if qk_head_dim != v_head_dim or qk_head_dim not in spec.head_dims:
        return (
            None,
            f"head dims (qk={qk_head_dim}, v={v_head_dim}) must be equal and in {spec.head_dims}",
        )
    sizevars = V.graph.sizevars
    if not all(
        sizevars.statically_known_equals(n.get_stride()[-1], 1)
        for n in (query, key, value)
    ):
        return None, "the head dim of Q, K and V must be contiguous"
    if not all(_offsets_fit_int32(n) for n in (query, key, value)):
        return None, "a (batch, head) slice of Q, K or V exceeds 32-bit offsets"
    num_programs = min(_num_programs(spec, c) for c in spec.configs)
    if not sizevars.guard_or_false(sympy.Le(query.get_size()[1], num_programs)):
        return None, f"more query heads than the {num_programs} persistent programs"
    if kernel_options.get("USE_TMA"):
        return None, "USE_TMA is not supported"
    if not any(_matches_forced_options(c, kernel_options) for c in spec.configs):
        forced = {k: kernel_options.get(k) for k in _FORCEABLE_OPTIONS}
        return None, f"forced {forced} matches none of {spec.configs}"
    if spec.check is not None and (
        reason := spec.check(query, key, value, kernel_options)
    ):
        return None, reason
    return spec, ""


def append_gluon_flex_choices(
    choices: list[Any],
    query,
    key,
    value,
    kernel_options: dict[str, Any],
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    score_mod_is_identity: bool,
    **choice_kwargs: Any,
) -> str | None:
    """Append Gluon forward choices. If none can be appended, return the reason
    and leave `choices` unchanged."""
    spec, reason = _get_gluon_spec(query, key, value, kernel_options)
    if spec is None:
        return reason

    num_choices = len(choices)

    for conf in spec.configs:
        options = {
            k: v
            for k, v in kernel_options.items()
            if not k.startswith(("fwd_", "bwd_"))
        }
        options.update(
            {k[4:]: v for k, v in kernel_options.items() if k.startswith("fwd_")}
        )
        if not _matches_forced_options(conf, options):
            continue
        if sparse_q_block_size % conf.block_m or sparse_kv_block_size % conf.block_n:
            continue
        options["BLOCK_M"] = conf.block_m
        options["BLOCK_N"] = conf.block_n
        options["num_warps"] = conf.num_warps
        options.setdefault("num_stages", 1)
        options["USE_TMA"] = False
        options["SPARSE_Q_BLOCK_SIZE"] = sparse_q_block_size
        options["SPARSE_KV_BLOCK_SIZE"] = sparse_kv_block_size
        options["NUM_PROGRAMS"] = _num_programs(spec, conf)
        options["NUM_XCDS"] = spec.num_xcds
        options["SCORE_MOD_IS_IDENTITY"] = score_mod_is_identity
        error = spec.template.maybe_append_choice(
            choices=choices, **choice_kwargs, **options
        )
        if error is not None:
            log.info("Gluon flex attention config %s rejected: %s", conf, error)
    if len(choices) == num_choices:
        return "no Gluon config is valid for these inputs"
    return None


def select_gluon_flex_choice(
    choices: list[Any], layout, input_gen_fns: dict[int, Any]
) -> TensorBox | None:
    """Autotune the Gluon choices, or warn and return None if none of them
    compiles so the caller builds Triton choices instead."""
    try:
        if len(choices) == 1 or config.deterministic:
            # autotune_select_algorithm picks choices[0] without compiling it.
            choices[0].precompile()
    except Exception as e:
        reason = f"its Gluon kernel failed to compile ({e})"
    else:
        try:
            out, _ = autotune_select_algorithm(
                "flex_attention",
                choices,
                list(choices[0].input_nodes),
                layout,
                input_gen_fns=input_gen_fns,
            )
            return out
        except NoValidChoicesError:
            reason = "every Gluon config failed to compile or benchmark"
    warn_gluon_fallback(reason)
    return None


def warn_gluon_fallback(reason: str) -> None:
    warning_once(
        log,
        "BACKEND='GLUON' is unavailable because %s; using the Triton template.",
        reason,
    )
