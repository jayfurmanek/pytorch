from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, TYPE_CHECKING

import torch

from ..select_algorithm import TritonTemplate, TritonTemplateKernel
from ..utils import IndentedBuffer
from .triton import TritonKernelOverrides, TritonOverrides


if TYPE_CHECKING:
    from collections.abc import Callable

    from ..scheduler import BaseSchedulerNode, Scheduler


fusion_log = torch._logging.getArtifactLogger(__name__, "fusion")


# Gluon rejects these tl ops because they take no layout. Their gl
# counterparts default to AutoLayout, which Gluon resolves against the
# template's concrete layouts.
_TL_LAYOUT_OPS = re.compile(r"\btl\.(full|zeros_like|zeros|arange)\(")


def triton_to_gluon(code: str) -> str:
    """Rewrite Inductor-generated Triton code so it is valid inside @gluon.jit."""
    return _TL_LAYOUT_OPS.sub(r"gl.\1(", code)


class GluonKernelOverrides(TritonKernelOverrides):
    # TritonKernelOverrides shapes constants to the kernel's range-tree rank,
    # which is 1 for templates. Triton broadcasts that against the template's
    # 2D tiles; Gluon requires equal ranks, so emit rank-0 constants.
    constant = TritonOverrides.constant


class GluonTemplateKernel(TritonTemplateKernel):
    """Renders a hand-written Gluon template.

    Code that Inductor generates into the template (score/mask modifications,
    store_output, size()) comes from the Triton codegen and is passed through
    triton_to_gluon.
    """

    jit_decorator = "@gluon.jit"
    overrides = GluonKernelOverrides  # type: ignore[assignment]

    @classmethod
    @lru_cache(None)
    def gen_common_triton_imports(cls) -> str:
        imports = IndentedBuffer()
        imports.splice(super().gen_common_triton_imports())
        imports.splice(
            """
            from triton.experimental import gluon
            from triton.experimental.gluon import language as gl
            """
        )
        return imports.getvalue()

    def size(self, name: str | None, index: int) -> str:
        return triton_to_gluon(super().size(name, index))

    def modification(self, *args: Any, **kwargs: Any) -> str:
        return triton_to_gluon(super().modification(*args, **kwargs))

    def _make_codegen_hook(
        self, subgraph_name: str, indent_width: int = 0
    ) -> Callable[[], str]:
        hook = super()._make_codegen_hook(subgraph_name, indent_width)
        return lambda: triton_to_gluon(hook())


# Fusing these into the Gluon flex attention template measured 25-30% slower
# than running them as a separate kernel (cos/sin on MI350X), while cheap
# epilogues (relu, casts, affine, residual add) were 5-9% faster.
_COSTLY_EPILOGUE_OPS = (
    "cos", "sin", "tan", "acos", "asin", "atan", "atan2",
    "cosh", "sinh", "tanh", "acosh", "asinh", "atanh",
    "exp", "exp2", "expm1", "log", "log2", "log10", "log1p",
    "erf", "erfc", "erfinv", "pow", "lgamma", "digamma",
)  # fmt: skip


def check_gluon_epilogue_fusion(
    scheduler: Scheduler, template: BaseSchedulerNode, epilogue: BaseSchedulerNode
) -> bool:
    """Fuse an epilogue only if it has no costly ops and the fused kernel
    compiles. Template selection's compile fallback only covers the unfused
    kernel, so a fused kernel that fails to compile must be caught here.
    Timing the fusion instead (benchmark_fusion) is not representative for
    flex: its benchmark inputs zero the block mask."""
    from ..scheduler import SchedulerNode

    for node in epilogue.get_nodes():
        if isinstance(node, SchedulerNode) and any(
            node._body.has_op(op) for op in _COSTLY_EPILOGUE_OPS
        ):
            fusion_log.debug("not fusing costly epilogue %s", node.get_name())
            return False
    try:
        future, mod = scheduler.compile_kernel(
            [*template.get_nodes(), *epilogue.get_nodes()]
        )
        if future is not None:
            future.result()
        else:
            mod.triton_.precompile()
    except Exception as e:
        fusion_log.debug("not fusing epilogue that fails to compile: %s", e)
        return False
    return True


class GluonTemplate(TritonTemplate):
    """A TritonTemplate whose source is a Gluon kernel.

    Epilogue fusion follows config.epilogue_fusion, subject to
    check_gluon_epilogue_fusion.
    """

    kernel_type = GluonTemplateKernel
    epilogue_fusion_check = staticmethod(check_gluon_epilogue_fusion)
