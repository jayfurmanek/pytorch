# Owner(s): ["module: inductor"]
import unittest
from unittest import mock

import torch
import torch._inductor.codegen.gluon_template as gluon_template
import torch._inductor.kernel.flex.flex_gluon as flex_gluon
from torch._inductor import config as inductor_config, select_algorithm
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code, run_fw_bw_and_get_code
from torch._logging._internal import warning_once
from torch.nn.attention.flex_attention import (
    and_masks,
    AuxRequest,
    create_block_mask,
    flex_attention,
)
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.inductor_utils import HAS_GPU_AND_TRITON
from torch.utils._triton import has_gluon


def _gluon_device() -> bool:
    if not (HAS_GPU_AND_TRITON and torch.version.hip and has_gluon()):
        return False
    arch = torch.cuda.get_device_properties().gcnArchName.split(":")[0]
    return arch in flex_gluon.GLUON_FLEX_FWD_TEMPLATES


requires_gluon_template = unittest.skipUnless(
    _gluon_device(), "requires a GPU arch with a Gluon flex template"
)
FLEX_LOGGER = "torch._inductor.kernel.flex.flex_attention"


def _causal(b, h, m, n):
    return m >= n


def _sliding_window(b, h, m, n):
    return (m >= n) & (m - n <= 256)


def _prefix_lm(b, h, m, n):
    return (n <= 128) | (m >= n)


def _times_two(score, b, h, m, n):
    return score * 2


def _rel_bias(score, b, h, m, n):
    return score + (m - n) * 0.01


def _softcap(score, b, h, m, n):
    return 20 * torch.tanh(score / 20)


def _alibi(slopes):
    def score_mod(score, b, h, m, n):
        return score + slopes[h] * (n - m)

    return score_mod


def _document_mask(doc_ids):
    def mask_mod(b, h, m, n):
        return doc_ids[m] == doc_ids[n]

    return mask_mod


def _compile_flex(backend, dynamic=False, **kwargs):
    def fn(q, k, v):
        return flex_attention(q, k, v, kernel_options={"BACKEND": backend}, **kwargs)

    return torch.compile(fn, dynamic=dynamic)


def _flatten(result):
    if isinstance(result, tuple):
        out, aux = result
        return [out, aux.lse, aux.max_scores]
    return [result]


def _max_err(x, ref):
    return (x.float() - ref.float()).abs().max().item()


class TestFlexGluon(InductorTestCase):
    def setUp(self):
        super().setUp()
        warning_once.cache_clear()

    def _make_qkv(self, device, dtype, B=2, Hq=4, Hkv=4, S_q=1024, S_kv=1024, D=128):
        torch.manual_seed(0)
        q = torch.randn(B, Hq, S_q, D, device=device, dtype=dtype)
        k = torch.randn(B, Hkv, S_kv, D, device=device, dtype=dtype)
        v = torch.randn(B, Hkv, S_kv, D, device=device, dtype=dtype)
        return q, k, v

    def _gluon_vs_triton(
        self, q, k, v, score_mod=None, mask_mod=None, dynamic=False, **kwargs
    ):
        """Gluon's error against an fp32 eager reference must be within a small
        factor of Triton's, and the Gluon kernel must actually have run."""
        B, Hq, S_q, _ = q.shape
        S_kv = k.shape[2]
        kwargs["enable_gqa"] = Hq != k.shape[1]
        kwargs["score_mod"] = score_mod
        if mask_mod is not None:
            kwargs["block_mask"] = create_block_mask(
                mask_mod, B, Hq, S_q, S_kv, device=q.device
            )
        ref = flex_attention(q.float(), k.float(), v.float(), **kwargs)
        gluon_out, (code, *_) = run_and_get_code(
            _compile_flex("GLUON", dynamic, **kwargs), q, k, v
        )
        triton_out = _compile_flex("TRITON", dynamic, **kwargs)(q, k, v)
        self.assertIn("@gluon.jit", code)

        outs = [_flatten(o) for o in (gluon_out, triton_out, ref)]
        for g, t, r in zip(*outs, strict=True):
            self.assertFalse(torch.isnan(g).any())
            atol = 2 * (r.float() + 0.3 - 0.3 - r.float()).abs().max().item()
            self.assertLessEqual(_max_err(g, r), 2 * _max_err(t, r) + atol)
        return code

    @requires_gluon_template
    @dtypes(torch.bfloat16, torch.float16)
    @parametrize("head_dim", [64, 128])
    def test_causal(self, device, dtype, head_dim):
        q, k, v = self._make_qkv(device, dtype, D=head_dim)
        self._gluon_vs_triton(q, k, v, mask_mod=_causal)

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    @parametrize("score_mod_name", ["times_two", "rel_bias", "softcap", "alibi"])
    def test_score_mod(self, device, dtype, score_mod_name):
        q, k, v = self._make_qkv(device, dtype)
        slopes = torch.linspace(-1, -0.1, q.shape[1], device=device)
        score_mod = {
            "times_two": _times_two,
            "rel_bias": _rel_bias,
            "softcap": _softcap,
            "alibi": _alibi(slopes),
        }[score_mod_name]
        self._gluon_vs_triton(q, k, v, score_mod=score_mod, mask_mod=_causal)

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    @parametrize("mask_name", ["none", "sliding_window", "prefix_lm", "document"])
    def test_mask_mod(self, device, dtype, mask_name):
        q, k, v = self._make_qkv(device, dtype)
        doc_ids = torch.arange(q.shape[2], device=device) // 300
        mask_mod = {
            "none": None,
            "sliding_window": _sliding_window,
            "prefix_lm": _prefix_lm,
            "document": and_masks(_causal, _document_mask(doc_ids)),
        }[mask_name]
        self._gluon_vs_triton(q, k, v, mask_mod=mask_mod)

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    @parametrize(
        "shape",
        [(1000, 1000), (512, 1024), (1024, 333)],
        name_fn=lambda s: f"{s[0]}x{s[1]}",
    )
    def test_non_divisible(self, device, dtype, shape):
        q, k, v = self._make_qkv(device, dtype, S_q=shape[0], S_kv=shape[1])
        self._gluon_vs_triton(q, k, v, score_mod=_rel_bias)

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    def test_gqa(self, device, dtype):
        q, k, v = self._make_qkv(device, dtype, Hq=8, Hkv=2)
        self._gluon_vs_triton(q, k, v, mask_mod=_causal)

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    def test_kv_batch_broadcast(self, device, dtype):
        q, k, v = self._make_qkv(device, dtype)
        self._gluon_vs_triton(q, k[:1], v[:1], mask_mod=_causal)

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    @parametrize(
        "shape",
        [(1, 1, 256), (8, 64, 512), (3, 5, 777)],
        name_fn=lambda s: "B{}_H{}_S{}".format(*s),
    )
    def test_persistent_schedule(self, device, dtype, shape):
        # Fewer work items than programs, many more, and an uneven split across
        # batch groups and query slots.
        B, H, S = shape
        q, k, v = self._make_qkv(device, dtype, B=B, Hq=H, Hkv=H, S_q=S, S_kv=S, D=64)
        self._gluon_vs_triton(q, k, v, mask_mod=_causal)

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    @parametrize("score_mod_name", ["none", "rel_bias"])
    def test_negative_scale(self, device, dtype, score_mod_name):
        # The softmax scales the row max instead of each score, which is only
        # exact for a positive scale.
        q, k, v = self._make_qkv(device, dtype, S_q=512, S_kv=512)
        mod = {"none": None, "rel_bias": _rel_bias}[score_mod_name]
        self._gluon_vs_triton(q, k, v, score_mod=mod, mask_mod=_causal, scale=-0.1)

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    def test_blocks_are_contiguous(self, device, dtype):
        q, k, v = self._make_qkv(device, dtype)
        block_mask = create_block_mask(_causal, 2, 4, 1024, 1024, device=device)
        options = {"BLOCKS_ARE_CONTIGUOUS": True}

        def fn(backend):
            kernel_options = {"BACKEND": backend, **options}
            return torch.compile(
                lambda q, k, v: flex_attention(
                    q, k, v, block_mask=block_mask, kernel_options=kernel_options
                )
            )

        out, (code, *_) = run_and_get_code(fn("GLUON"), q, k, v)
        self.assertIn("@gluon.jit", code)
        self.assertEqual(out, fn("TRITON")(q, k, v), atol=2e-2, rtol=2e-2)

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    def test_lse_and_max(self, device, dtype):
        q, k, v = self._make_qkv(device, dtype, S_q=1000, S_kv=1000)
        self._gluon_vs_triton(
            q, k, v, mask_mod=_causal, return_aux=AuxRequest(lse=True, max_scores=True)
        )

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    def test_prescale_qk(self, device, dtype):
        q, k, v = self._make_qkv(device, dtype)

        def fn(backend):
            options = {"BACKEND": backend, "PRESCALE_QK": True}
            return torch.compile(
                lambda q, k, v: flex_attention(q, k, v, kernel_options=options)
            )

        out, (code, *_) = run_and_get_code(fn("GLUON"), q, k, v)
        self.assertIn("@gluon.jit", code)
        self.assertEqual(out, fn("TRITON")(q, k, v), atol=2e-2, rtol=2e-2)

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    def test_dynamic_shapes(self, device, dtype):
        for seq_len in (512, 1000):
            q, k, v = self._make_qkv(device, dtype, S_q=seq_len, S_kv=seq_len)
            self._gluon_vs_triton(q, k, v, score_mod=_rel_bias, dynamic=True)

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    def test_backward_uses_triton(self, device, dtype):
        q, k, v = (
            t.requires_grad_() for t in self._make_qkv(device, dtype, S_q=512, S_kv=512)
        )
        block_mask = create_block_mask(_causal, 2, 4, 512, 512, device=device)

        def run(backend):
            fn = _compile_flex(backend, score_mod=_softcap, block_mask=block_mask)
            out, codes = run_fw_bw_and_get_code(lambda: fn(q, k, v).sum())
            grads = torch.autograd.grad(fn(q, k, v).sum(), (q, k, v))
            return grads, "\n".join(codes)

        gluon_grads, code = run("GLUON")
        triton_grads, _ = run("TRITON")
        self.assertIn("@gluon.jit", code)
        for g, t in zip(gluon_grads, triton_grads, strict=True):
            self.assertEqual(g, t, atol=2e-2, rtol=2e-2)

    def _assert_falls_back(self, q, k, v, reason, **kwargs):
        with self.assertLogs(FLEX_LOGGER, level="WARNING") as logs:
            out, (code, *_) = run_and_get_code(
                _compile_flex("GLUON", **kwargs), q, k, v
            )
        self.assertNotIn("@gluon.jit", code)
        self.assertIn(reason, "\n".join(logs.output))
        self.assertEqual(out, _compile_flex("TRITON", **kwargs)(q, k, v))

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @dtypes(torch.bfloat16)
    def test_fallback_no_template_for_arch(self, device, dtype):
        q, k, v = self._make_qkv(device, dtype, S_q=256, S_kv=256)
        with mock.patch.dict(flex_gluon.GLUON_FLEX_FWD_TEMPLATES, clear=True):
            reason = (
                "no Gluon template"
                if torch.version.hip and has_gluon()
                else "requires ROCm"
            )
            self._assert_falls_back(q, k, v, reason)

    def _auto_choices(self, q, k, v):
        """Compile BACKEND="AUTO" and return the output, the generated code and
        the descriptions of the choices that were benchmarked."""
        descriptions = []

        def saver(timings, name, *args):
            if name == "flex_attention":
                descriptions.extend(c.description for c in timings)

        select_algorithm.add_feedback_saver(saver)
        try:
            out, (code, *_) = run_and_get_code(_compile_flex("AUTO"), q, k, v)
        finally:
            select_algorithm.clear_feedback_savers()
        return out, code, descriptions

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    @parametrize("enabled", [False, True])
    def test_auto_autotunes_gluon(self, device, dtype, enabled):
        q, k, v = self._make_qkv(device, dtype, S_q=256, S_kv=256, D=64)
        with inductor_config.patch(max_autotune=True, max_autotune_flex_gluon=enabled):
            out, _, descriptions = self._auto_choices(q, k, v)
        num_gluon = sum("NUM_PROGRAMS" in d for d in descriptions)
        self.assertEqual(num_gluon, 2 if enabled else 0)
        self.assertGreater(len(descriptions), num_gluon)
        self.assertEqual(out, _compile_flex("TRITON")(q, k, v), atol=2e-2, rtol=2e-2)

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    @parametrize("case", ["unsupported", "compile_failure"])
    def test_auto_gluon_is_silent(self, device, dtype, case):
        # Unlike BACKEND="GLUON", AUTO never warns about Gluon: it only adds
        # choices when they are usable, and a failed compile loses autotuning.
        D = 32 if case == "unsupported" else 64
        q, k, v = self._make_qkv(device, dtype, S_q=256, S_kv=256, D=D)
        to_gluon = gluon_template.triton_to_gluon
        broken_kernels = []

        def broken(code):
            code = to_gluon(code)
            if "post_mod_scores" not in code:
                return code
            broken_kernels.append(code)
            return f"{code}\nundefined_name"

        with (
            inductor_config.patch(max_autotune=True, max_autotune_flex_gluon=True),
            mock.patch.object(gluon_template, "triton_to_gluon", broken),
            self.assertNoLogs(FLEX_LOGGER, level="WARNING"),
        ):
            out, code, descriptions = self._auto_choices(q, k, v)
        self.assertNotIn("@gluon.jit", code)
        self.assertFalse(any("NUM_PROGRAMS" in d for d in descriptions))
        self.assertEqual(bool(broken_kernels), case == "compile_failure")
        self.assertEqual(out, _compile_flex("TRITON")(q, k, v), atol=2e-2, rtol=2e-2)

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    @parametrize("case", ["relu", "cos", "compile_failure"])
    def test_epilogue_fusion(self, device, dtype, case):
        # Cheap epilogues fuse; costly ops, and fused kernels that fail to
        # compile, run as a separate kernel instead.
        q, k, v = self._make_qkv(device, dtype, S_q=256, S_kv=256)
        epilogue = torch.cos if case == "cos" else torch.relu
        to_gluon = gluon_template.triton_to_gluon

        def broken(code):
            code = to_gluon(code)
            return f"{code}\nundefined_name" if "maximum" in code else code

        def fn(backend):
            options = {"BACKEND": backend}
            return torch.compile(
                lambda q, k, v: epilogue(
                    flex_attention(q, k, v, kernel_options=options)
                )
            )

        with mock.patch.object(
            gluon_template,
            "triton_to_gluon",
            broken if case == "compile_failure" else to_gluon,
        ):
            out, (code, *_) = run_and_get_code(fn("GLUON"), q, k, v)
        self.assertIn("@gluon.jit", code)
        separate_kernel = f"triton_poi_fused_{epilogue.__name__}"
        self.assertEqual(separate_kernel in code, case != "relu")
        self.assertEqual(out, fn("TRITON")(q, k, v), atol=2e-2, rtol=2e-2)

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @dtypes(torch.bfloat16)
    def test_short_query_uses_triton_decode(self, device, dtype):
        q, k, v = self._make_qkv(device, dtype, S_q=16, S_kv=1024)
        with self.assertLogs(FLEX_LOGGER, level="WARNING") as logs:
            out, (code, *_) = run_and_get_code(_compile_flex("GLUON"), q, k, v)
        self.assertNotIn("@gluon.jit", code)
        self.assertIn("SPLIT_KV", code)
        self.assertIn("no decode template", "\n".join(logs.output))
        self.assertEqual(out, _compile_flex("AUTO")(q, k, v))

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    def test_fallback_head_dim(self, device, dtype):
        q, k, v = self._make_qkv(device, dtype, S_q=256, S_kv=256, D=32)
        self._assert_falls_back(q, k, v, "head dims")

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    def test_fallback_more_heads_than_programs(self, device, dtype):
        q, k, v = self._make_qkv(device, dtype, Hq=16, Hkv=16, S_q=256, S_kv=256)
        with mock.patch.object(flex_gluon, "_num_programs", return_value=8):
            self._assert_falls_back(q, k, v, "more query heads than the 8")

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    @parametrize("forced_block_n", [None, 64])
    def test_fallback_compile_failure(self, device, dtype, forced_block_n):
        # Two configs go through autotuning; a forced BLOCK_N leaves one, which
        # autotuning would return without compiling.
        q, k, v = self._make_qkv(device, dtype, S_q=256, S_kv=256)
        to_gluon = gluon_template.triton_to_gluon

        def broken(code):
            code = to_gluon(code)
            return f"{code}\nundefined_name" if "post_mod_scores" in code else code

        options = {"BACKEND": "GLUON"}
        if forced_block_n is not None:
            options["BLOCK_N"] = forced_block_n
        fn = torch.compile(
            lambda q, k, v: flex_attention(q, k, v, kernel_options=options)
        )
        with (
            mock.patch.object(gluon_template, "triton_to_gluon", broken),
            self.assertLogs(FLEX_LOGGER, level="WARNING") as logs,
        ):
            out, (code, *_) = run_and_get_code(fn, q, k, v)
        self.assertNotIn("@gluon.jit", code)
        self.assertIn("failed to compile", "\n".join(logs.output))
        self.assertEqual(out, _compile_flex("TRITON")(q, k, v))

    @requires_gluon_template
    @dtypes(torch.bfloat16)
    def test_fallback_forced_block_size(self, device, dtype):
        q, k, v = self._make_qkv(device, dtype, S_q=256, S_kv=256)

        def fn(backend):
            options = {"BACKEND": backend, "BLOCK_M": 64, "BLOCK_N": 64}
            return torch.compile(
                lambda q, k, v: flex_attention(q, k, v, kernel_options=options)
            )

        with self.assertLogs(FLEX_LOGGER, level="WARNING") as logs:
            out, (code, *_) = run_and_get_code(fn("GLUON"), q, k, v)
        self.assertNotIn("@gluon.jit", code)
        self.assertIn("matches none of", "\n".join(logs.output))
        self.assertEqual(out, fn("TRITON")(q, k, v))


instantiate_device_type_tests(TestFlexGluon, globals(), only_for="cuda")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
