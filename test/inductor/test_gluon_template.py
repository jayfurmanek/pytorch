# Owner(s): ["module: inductor"]
import contextlib
import unittest
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.codegen.gluon_template import GluonTemplate, triton_to_gluon
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.ir import FixedLayout
from torch._inductor.kernel.flex.common import (
    build_subgraph_buffer,
    create_placeholder,
    freeze_irnodes,
    maybe_realize,
)
from torch._inductor.lowering import _register_lowering, empty_strided, lowerings
from torch._inductor.select_algorithm import autotune_select_algorithm, SymbolicGridFn
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.nn.attention.flex_attention import flex_attention
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON
from torch.utils._triton import has_gluon


@SymbolicGridFn
def toy_grid(B, H, M, D, meta, *, cdiv):
    return (B * H, cdiv(M, meta["BLOCK_M"]), 1)


# out[b, h, m, n] = score_mod(q[b, h, m, n], b, h, m, n): flex's modification
# hook without the attention math, so score_mod codegen is tested in isolation.
toy_template = GluonTemplate(
    name="gluon_toy_score_mod",
    grid=toy_grid,
    source=r"""
{{def_kernel("Q")}}
    stride_qz, stride_qh, stride_qm, stride_qk = {{stride("Q")}}
    Q_LEN = {{size("Q", 2)}}
    H = {{size("Q", 1)}}
    L: gl.constexpr = gl.BlockedLayout([1, 1], [1, WARP_SIZE], [gl.num_warps(), 1], [1, 0])

    off_z = gl.program_id(0) // H
    off_h = gl.program_id(0) % H
    offs_m = gl.program_id(1) * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, L))
    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, L))
    m = offs_m[:, None]
    n = offs_n[None, :]
    mask = (m < Q_LEN) & (n < BLOCK_N)
    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + m * stride_qm + n * stride_qk
    q = gl.load(q_ptrs, mask=mask, other=0.0).to(gl.float32)
    # Like flex's get_bounded_indices: score_mod gathers are unmasked.
    m_bounded = m % Q_LEN

    {{ modification(
        subgraph_number=0,
        output_name="post_mod",
        score="q",
        b="off_z",
        h="off_h",
        m="m_bounded",
        n="n",
        input_shapes={
            "score": ("BLOCK_M", "BLOCK_N"),
            "m": ("BLOCK_M", "1"),
            "n": ("1", "BLOCK_N"),
        },
        input_dtypes={"b": "index", "h": "index", "m": "index", "n": "index"},
    ) | indent_except_first(1) }}

    {{store_output(
        ("off_z", "off_h", "m", "n"),
        "post_mod",
        mask="mask",
        val_shape=("BLOCK_M", "BLOCK_N"),
    )}}
""",
)


def toy_flex_lowering(
    query,
    key,
    value,
    subgraph,
    block_mask,
    scale,
    kernel_options,
    score_mod_other_buffers,
    mask_mod_other_buffers,
):
    device = query.get_device()
    placeholders = [
        create_placeholder(name, dtype, device)
        for name, dtype in [
            ("score", query.get_dtype()),
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]
    score_mod_other_buffers = maybe_realize(score_mod_other_buffers)
    freeze_irnodes(score_mod_other_buffers)
    subgraph_buffer = build_subgraph_buffer(
        placeholders + list(score_mod_other_buffers), subgraph
    )
    freeze_irnodes(subgraph_buffer)
    (query,) = maybe_realize([query])

    B, H, M, D = query.get_size()
    layout = FixedLayout(device, query.get_dtype(), [B, H, M, D], query.get_stride())
    choices: list = []
    error = toy_template.maybe_append_choice(
        choices,
        input_nodes=[query],
        layout=layout,
        subgraphs=[subgraph_buffer],
        num_stages=1,
        num_warps=4,
        BLOCK_M=32,
        BLOCK_N=D,
        WARP_SIZE=torch.cuda.get_device_properties(device).warp_size,
    )
    if error is not None:
        raise error
    out, _ = autotune_select_algorithm("gluon_toy", choices, [query], layout)
    lse = empty_strided([B, H, M], None, dtype=torch.float32, device=device)
    max_scores = empty_strided([B, H, M], None, dtype=torch.float32, device=device)
    return out, lse, max_scores


@contextlib.contextmanager
def use_toy_flex_lowering():
    hop = torch.ops.higher_order.flex_attention
    with mock.patch.dict(lowerings):
        _register_lowering(
            hop,
            toy_flex_lowering,
            broadcast=False,
            type_promotion_kind=None,
            convert_input_to_bool=False,
            lowering_dict=lowerings,
        )
        yield


B, H, M, D = 2, 4, 200, 64


def make_captures(device):
    g = torch.Generator(device=device).manual_seed(0)
    return {
        "head_bias": torch.randn(H, device=device, generator=g),
        "bias_2d": torch.randn(M, D, device=device, generator=g),
        "doc_q": torch.randint(0, 3, (M,), device=device, generator=g),
        "doc_n": torch.randint(0, 3, (D,), device=device, generator=g),
        "doc_bias": torch.randn(3, 3, device=device, generator=g),
    }


SCORE_MODS = {
    "constant": lambda c: lambda s, b, h, m, n: s * 2.0 + 1,
    "relative": lambda c: lambda s, b, h, m, n: s + (m - n),
    "alibi": lambda c: lambda s, b, h, m, n: s + c["head_bias"][h] * (n - m),
    "bias_2d": lambda c: lambda s, b, h, m, n: s + c["bias_2d"][m, n],
    "softcap": lambda c: lambda s, b, h, m, n: 30 * torch.tanh(s / 30),
    "causal_where": lambda c: lambda s, b, h, m, n: torch.where(m >= n, s, -1e4),
    "doc_gather": lambda c: lambda s, b, h, m, n: (
        s + c["doc_bias"][c["doc_q"][m], c["doc_n"][n]]
    ),
}


def reference(score_mod, q):
    idx = [torch.arange(s, device=q.device) for s in q.shape]
    b, h, m, n = torch.meshgrid(*idx, indexing="ij")
    return score_mod(q.float(), b, h, m, n).to(q.dtype)


@unittest.skipUnless(HAS_GPU_AND_TRITON and has_gluon(), "requires gpu and gluon")
@instantiate_parametrized_tests
class TestGluonTemplate(TestCase):
    def _run(self, score_mod, dynamic=False):
        q = torch.randn(B, H, M, D, device=GPU_TYPE, dtype=torch.bfloat16)
        if dynamic:
            torch._dynamo.mark_dynamic(q, 2)

        def fn(q):
            return flex_attention(q, q, q, score_mod=score_mod)

        with use_toy_flex_lowering():
            out, (code, *_) = run_and_get_code(torch.compile(fn, fullgraph=True), q)
        self.assertEqual(out, reference(score_mod, q))
        self.assertIn("@gluon.jit", code)
        return code

    @parametrize("name", list(SCORE_MODS))
    def test_score_mod(self, name):
        self._run(SCORE_MODS[name](make_captures(GPU_TYPE)))

    @parametrize("name", ["relative", "bias_2d"])
    def test_dynamic_seq_len(self, name):
        self._run(SCORE_MODS[name](make_captures(GPU_TYPE)), dynamic=True)

    @parametrize("name", ["constant", "alibi"])
    def test_int64_indexing(self, name):
        with mock.patch.object(
            TritonScheduling, "can_use_32bit_indexing", return_value=False
        ):
            code = self._run(SCORE_MODS[name](make_captures(GPU_TYPE)))
        self.assertIn("INDEX_DTYPE : tl.constexpr = tl.int64", code)
        self.assertIn(f"Q_LEN = gl.full([], {M}, dtype=INDEX_DTYPE)", code)

    @parametrize(
        "allow_epilogue_fusion",
        [False, None],
        name_fn=lambda v: "disallowed" if v is False else "config_default",
    )
    @config.patch(epilogue_fusion=True, fx_graph_cache=False)
    def test_epilogue_fusion(self, allow_epilogue_fusion):
        score_mod = SCORE_MODS["constant"](None)
        q = torch.randn(B, H, M, D, device=GPU_TYPE, dtype=torch.bfloat16)

        def fn(q):
            return flex_attention(q, q, q, score_mod=score_mod).relu()

        with (
            use_toy_flex_lowering(),
            mock.patch.object(
                GluonTemplate, "allow_epilogue_fusion", allow_epilogue_fusion
            ),
        ):
            out, (code, *_) = run_and_get_code(torch.compile(fn, fullgraph=True), q)
        self.assertEqual(out, reference(score_mod, q).relu())
        # Disallowing epilogue fusion keeps the relu in its own kernel.
        self.assertEqual(
            "triton_poi_fused_relu" in code, allow_epilogue_fusion is False
        )


class TestTritonToGluon(TestCase):
    def test_rewrites_layout_ops(self):
        code = (
            "tl.full([], 1.0, tl.float32) + tl.zeros_like(x) + tl.zeros([1], tl.int32)"
        )
        expected = (
            "gl.full([], 1.0, tl.float32) + gl.zeros_like(x) + gl.zeros([1], tl.int32)"
        )
        self.assertEqual(triton_to_gluon(code), expected)
        self.assertEqual(triton_to_gluon("tl.arange(0, 4)"), "gl.arange(0, 4)")

    def test_leaves_other_ops(self):
        code = "tl.load(in_ptr0 + x0) + libdevice.tanh(tmp0) + my_tl.full(y)"
        self.assertEqual(triton_to_gluon(code), code)


if __name__ == "__main__":
    run_tests()
