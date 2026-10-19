import re
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from ..select_algorithm import TritonTemplate, TritonTemplateKernel
from ..utils import IndentedBuffer
from .triton import TritonKernelOverrides, TritonOverrides


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


class GluonTemplate(TritonTemplate):
    """A TritonTemplate whose source is a Gluon kernel.

    Epilogue fusion is disabled until fused epilogue codegen is validated
    under Gluon.
    """

    kernel_type = GluonTemplateKernel
    allow_epilogue_fusion = False
