# Gluon support in Inductor: plan

Status: draft. Based on `main` at `79f81f9e6d3` with Triton 3.8.0, which ships
`triton.experimental.gluon` with `amd` and `nvidia` backends. Nothing in this
checkout references Gluon yet.

## Background

Gluon kernels are `GluonJITFunction`, a subclass of `JITFunction`. They differ
from Triton kernels in three ways that matter to Inductor:

1. **Decorator and namespace.** They use `@gluon.jit` and `gl.*`, not
   `@triton.jit` and `tl.*`.
2. **Compile entry point.** Triton compiles them through `GluonASTSource` (which
   sets `language=GLUON`), not `ASTSource`. `GluonJITFunction` only sets
   `self.ASTSource` inside `create_binder()`, so callers should branch on
   `fn.is_gluon()` rather than rely on that attribute.
3. **IR level and architecture.** Gluon lowers straight to TTGIR-level ops with
   explicit layouts. Layouts are written into the source, so a kernel is tied to
   an architecture family (e.g. CDNA MFMA vs. NVIDIA MMA). The TTIR-walking
   mutation analysis does not understand these ops.

Inductor hard-codes the Triton path at these sites:

| Site | Assumption |
|---|---|
| `torch/_inductor/codegen/triton.py` `gen_common_triton_imports` | Emits `triton`/`tl` imports, plus a one-off `try: import tlx` |
| `torch/_inductor/codegen/wrapper.py` `write_header`, `write_triton_header_once` | Wrapper headers import `triton`/`tl` |
| `torch/_inductor/codegen/wrapper.py` `define_user_defined_triton_kernel` | User kernels are re-emitted under a literal `@triton.jit` |
| `torch/_inductor/codegen/wrapper.py` `user_defined_triton_kernel_transitive_closure_source_code` | Helper `JITFunction`s get `@triton.jit`; only `triton` and `tl` are treated as known globals |
| `torch/_inductor/runtime/triton_heuristics.py` `_precompile_config` | Always `ASTSource(self.fn, ...)` |
| `torch/_higher_order_ops/triton_kernel_wrap.py` `generate_ttir` | Always `ASTSource`, then a TTIR op walk |
| `torch/_inductor/select_algorithm.py` `jit_lines` | Templates are `@triton.jit` |

## Phase 1: generalize kernel-language imports (pure refactor)

- Replace the inline TLX `try/except` in `gen_common_triton_imports` with a small
  table of optional kernel-language extensions. Each entry is a frozen dataclass
  with:
  - the module to probe (e.g. `triton.language.extra.tlx`);
  - the import lines it contributes;
  - when to emit them: always when the module imports, or only when a kernel
    actually uses that language.
- Register two entries:
  - TLX, emitted whenever available. This keeps today's behavior exactly.
  - Gluon (`from triton.experimental import gluon`,
    `from triton.experimental.gluon import language as gl`), emitted only on use.
- Add `has_gluon()` to `torch/utils/_triton.py` next to the existing
  `has_triton_*` probes. Add `GluonASTSource` to
  `torch/_inductor/runtime/triton_compat.py`, `None` when unavailable.
- Route the wrapper headers and the user-kernel import path through the same
  helper, so the import list lives in one place.

**Why Gluon is emitted only on use:** import lines are part of the kernel source
and therefore the cache key. Adding Gluon imports to every kernel would
invalidate the FX and Triton caches fleet-wide and slow imports in compile
workers. The `tlx_mode_default` docstring in `torch/_inductor/config.py` makes
the same argument.

**Test plan:** generated code must not change. Existing expected-output tests in
`test/inductor` pass unchanged; run `test_triton_kernels.py` and
`test_torchinductor.py -k triton`.

## Phase 2: user-written Gluon kernels in `torch.compile`

The smallest step that is useful on its own.

- **Dynamo / HOP.** Check that `GluonJITFunction` and
  `triton.autotune(gluon_kernel)` trace through `TritonKernelVariable` and
  `triton_kernel_wrapper_mutation`. The subclass should pass the existing
  `isinstance(..., JITFunction)` checks in `triton_kernel_wrap.py`, but this is
  unverified.
- **Mutation analysis.** Skip TTIR analysis for Gluon kernels: treat every
  tensor argument as mutated and log why. This matches the result of the
  existing fallback, but avoids a doomed compile and a confusing "op we do not
  know how to analyze" warning. A precise TTGIR-level analysis can follow.
- **Codegen.** Emit `@gluon.jit` for the top-level kernel and for Gluon helpers
  in the transitive closure; treat `gluon`/`gl` like `triton`/`tl` there. Make
  sure layout constexprs (`gl.BlockedLayout`, `gl.amd.AMDMFMALayout`, ...)
  survive a repr round-trip through `get_importable_constexpr_types`, whether
  passed as arguments or captured as globals.
- **Runtime.** In `_precompile_config`, use `GluonASTSource` when
  `self.fn.is_gluon()`. Confirm that pickling the kernel to compile workers keeps
  the subclass.
- **Architecture.** Nothing new is required: the user picks the arch-specific
  kernel in Python, Dynamo traces that choice, and compilation targets
  `DeviceProperties.cc` (see "Target architecture" below).
- **AOTInductor / cpp-wrapper.** These load compiled binaries independent of the
  source language. Add one AOTI test once the Python wrapper path works.

**Test plan:** device-generic tests in `test/inductor/test_triton_kernels.py`,
guarded by `@requires_gluon`. Kernels: a plain one, one with a helper, one with
`triton.autotune`. Run each under:

- `backend="eager"`, `"aot_eager"` and `"inductor"`;
- dynamic shapes;
- `config.triton.unique_user_kernel_names`;
- a mutated-output check.

Test kernels need per-target layouts: a CDNA MFMA/blocked layout (gfx950) and an
NVIDIA blocked layout for CUDA CI.

## Phase 3 (optional, separate RFC): Inductor-generated Gluon templates

- Let `TritonTemplate` and `jit_lines` take a kernel language, so Gluon
  templates get `@gluon.jit` and the Gluon imports.
- Register Gluon templates (e.g. an MFMA GEMM on gfx950) through the existing
  `heuristics/template` registry and `InductorChoices`, competing with Triton
  templates during autotuning.
- Add arch-keyed template registration (below), so templates for different
  architectures can coexist without each checking the arch itself.
- This is the in-tree, supported version of what TLX does out of tree by
  monkey-patching `TritonTemplate`; TLX could move onto it.
- Generating pointwise or reduction kernels in Gluon is out of scope.

## Target architecture

What exists today:

- `DeviceProperties.cc` (`torch/_inductor/runtime/hints.py`) is filled from
  `get_compute_capability()` (`torch/_dynamo/device_interface.py`): an int like
  `90` or `100` on CUDA, the bare `gcnArchName` like `"gfx950"` on ROCm. This
  matches Triton's `GPUTarget.arch`.
- It is baked into each kernel's `triton_meta["device"]`, so it is part of the
  generated source and cache key; kernels for different archs never share a
  cache entry.
- `_create_compile_meta` copies it to `compile_meta["cc"]`, and
  `_precompile_config` builds `GPUTarget(device_type, cc, warp_size)` from it,
  with a `cc_override` parameter.

What is missing:

- The template registry key is `(template_name, device_type, op_name)`
  (`torch/_inductor/heuristics/registry.py`), with no arch. Arch gating is ad
  hoc: `register=torch.version.hip is None` at import time, checks inside
  heuristic config generation, and scattered helpers
  (`is_datacenter_blackwell_arch`, `_rocm_gcn_arch`, `is_gfx1250_arch`,
  `_rocm_native_device_arch_name`).
- There is no standard normalized arch value at codegen time. On ROCm,
  `get_device_capability()` returns `(9, 5)` for gfx950, so CUDA-style checks
  like `>= (9, 0)` misfire.

Proposed:

- One cached helper returning the normalized target (backend, arch, warp size),
  backed by `DeviceProperties.create(device)` and safe in compile workers. All
  template and Gluon selection uses it instead of ad hoc property reads.
- Optional `arch` on `register_template_heuristic`, e.g.
  `register_template_heuristic("mm", "cuda", arch=("gfx950",))`, falling back to
  the device-only key when no arch-specific entry matches.
- Raise a clear error when a Gluon kernel is compiled for an arch it was not
  written for (e.g. via `cc_override` or AOTI cross-compilation), instead of
  surfacing a confusing compile failure.

## Sequencing and open questions

- Land Phase 1 and Phase 2 as separate PRs; Phase 1 changes no output.
- Keep TLX's import unconditional in Phase 1. Gating it on `tlx_mode` alters
  cache keys for TLX users and should be its own change.
- Open: is the goal user-written Gluon kernels inside `torch.compile`
  (Phases 1-2), Inductor-generated Gluon templates (Phase 3), or both?
