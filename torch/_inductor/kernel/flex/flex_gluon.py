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

from ...codegen.gluon_template import GluonTemplate
from ...select_algorithm import SymbolicGridFn
from .common import load_flex_template
from .flex_attention import log


if TYPE_CHECKING:
    from collections.abc import Callable


@dataclasses.dataclass(frozen=True)
class GluonFlexConfig:
    block_m: int
    block_n: int
    num_warps: int
    # Persistent programs launched per CU.
    programs_per_cu: int = 2


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
    configs=tuple(
        GluonFlexConfig(128, block_n, 4, programs_per_cu)
        for block_n in (64, 32)
        for programs_per_cu in (2, 3, 4)
    ),
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


def _num_programs(query, spec: GluonFlexTemplateSpec, conf: GluonFlexConfig) -> int:
    num_cus = torch.cuda.get_device_properties(query.get_device()).multi_processor_count
    return conf.programs_per_cu * num_cus // spec.num_xcds * spec.num_xcds


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
    num_programs = min(_num_programs(query, spec, c) for c in spec.configs)
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
) -> None:
    """Append Gluon forward choices, or warn and leave `choices` empty so the
    caller builds Triton choices instead."""
    spec, reason = _get_gluon_spec(query, key, value, kernel_options)
    if spec is None:
        warning_once(
            log,
            "BACKEND='GLUON' is unavailable because %s; using the Triton template.",
            reason,
        )
        return

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
        options["NUM_PROGRAMS"] = _num_programs(query, spec, conf)
        options["NUM_XCDS"] = spec.num_xcds
        options["SCORE_MOD_IS_IDENTITY"] = score_mod_is_identity
        error = spec.template.maybe_append_choice(
            choices=choices, **choice_kwargs, **options
        )
        if error is not None:
            log.info("Gluon flex attention config %s rejected: %s", conf, error)
    if not choices:
        warning_once(
            log,
            "BACKEND='GLUON' has no valid Gluon config for these inputs; using the Triton template.",
        )
