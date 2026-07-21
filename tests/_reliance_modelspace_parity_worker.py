"""Standalone worker for the numerical-parity test
(step3_modelspace_preextraction_gate_brief.md Item 2).

Run as a FRESH SUBPROCESS with OPENBLAS_NUM_THREADS/MKL_NUM_THREADS/
OMP_NUM_THREADS/NUMEXPR_NUM_THREADS already set to "1" in its environment
BEFORE this process's own `import numpy` runs -- BLAS reads these at
library-LOAD time, so a fresh process started this way initializes its
BLAS backend single-threaded natively (no threadpool_limits needed for
ITS OWN computation). Any subprocess workers this process itself spawns
(run_reliance_battery.py's own subprocess-isolated crossfit/rank-curve
tasks) inherit this SAME environment at their own spawn time, so they are
single-threaded too -- one mechanism, launched from the test, pins the
entire process tree (the "robust, order-independent way" the brief
describes, rather than needing threadpool_limits in the parent AND env
vars for the children as two separate mechanisms). Also applies
threadpoolctl.threadpool_limits(limits=1) around the computation as a
belt-and-suspenders measure, and reports threadpoolctl.threadpool_info()
plus the observed env vars as proof-of-pin evidence.

PRECISION, investigated and resolved (not assumed): run_reliance_battery.py's
own _write_battery_npz (line 539, `Z=np.asarray(Z, dtype=np.float32)`)
unconditionally casts Z to float32 before its subprocess workers ever see
it -- this is real, pre-existing, protected (run_reliance_battery.py must
stay byte-identical) behavior, independent of what dtype the caller
passes in. An EARLIER version of this comparison passed float64 Z to both
sides and measured a ~6.4e-6 max relative residual under a VERIFIED
single-thread pin -- well above the brief's ~1e-9 expectation. Isolated
with no subprocess involved at all (see this branch's commit history):
casting the SAME synthetic Z to float32 and re-running the identical
crossfit code reproduces the SAME ~5e-6-order residual on a near-zero
alignment value (true value ~4e-6; float32 rounding perturbs it by a
comparable ABSOLUTE amount, which is a huge RELATIVE error near zero) --
confirming the earlier residual was a float32-vs-float64 PRECISION
difference amplified by relative-error math near small values, not
subprocess/BLAS non-determinism and not divergent math between the two
implementations. This worker therefore casts Z to float32 BEFORE calling
EITHER side, matching what both paths actually operate on in real
production use anyway (scripts/extract_model_embeddings.py and this
sibling's own load_model_space_embeddings both cast to float32 at load;
run_battery's _write_battery_npz does too) -- a genuine apples-to-apples
comparison at the SAME precision, not an inflated one. Under that
precision-matched comparison, THIS worker measured EXACT (0.0) relative
residual on every continuous field, every fold, both estimators.

Prints ONE JSON blob to stdout: proof-of-pin evidence + every compared
field's value from both the model-space sibling (run_reliance_modelspace.
run_modelspace_battery/run_checkpoint_crossfit) and run_reliance_battery.py's
own run_battery (imported, unmodified), on IDENTICAL float32 synthetic
single-checkpoint data, identical seed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

_PIN_ENV_VARS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def main() -> None:
    import threadpoolctl

    from test_reliance_modelspace import _make_multi_checkpoint_synthetic_battery  # noqa: E402
    from run_reliance_battery import run_battery  # noqa: E402
    from run_reliance_modelspace import run_modelspace_battery  # noqa: E402

    env_vars = {v: os.environ.get(v) for v in _PIN_ENV_VARS}

    with threadpoolctl.threadpool_limits(limits=1):
        import numpy as np

        info = threadpoolctl.threadpool_info()
        blas_threads = [lib.get("num_threads") for lib in info if lib.get("user_api") == "blas"]

        Z_by_checkpoint, factor, y, groups, checkpoints = _make_multi_checkpoint_synthetic_battery(runs=("ckA",))
        run = "ckA"
        # Cast to float32 BEFORE either side computes anything -- matching
        # what run_battery's own _write_battery_npz does unconditionally
        # (and what real Phase B data is anyway, per extract_model_
        # embeddings.py's and load_model_space_embeddings' own float32
        # casts) -- see module docstring for why this is the correct,
        # apples-to-apples precision to compare at.
        Z32_by_checkpoint = {k: np.asarray(v, dtype=np.float32) for k, v in Z_by_checkpoint.items()}
        spec = dict(name="parity_test", corpus="diffssd", factor="generator_id", grouping="source_id")

        sibling_result = run_modelspace_battery(
            spec, Z32_by_checkpoint, factor, y, groups, checkpoints,
            ranks=[1, 2, 3], n_boot=0, seed=13, max_rows_per_level=None,
        )
        original_result = run_battery(
            spec, Z32_by_checkpoint[run], factor, y, groups, checkpoints,
            ranks=[1, 2, 3], n_boot=0, seed=13, layer_mode="fixed",
            w_metrics_enabled=True, max_rows_per_level=None,
        )

    residuals = []
    discrete_mismatches = []
    for estimator in ("lda", "probe"):
        sib_folds = sibling_result["estimators"][estimator]["fold_results"]
        orig_folds = original_result["estimators"][estimator]["fold_results"]
        for sib_fold, orig_fold in zip(sib_folds, orig_folds):
            sib_ck = sib_fold["effect"]["per_checkpoint"][run]
            orig_ck = orig_fold["effect"]["per_checkpoint"][run]

            for field, a, b in (
                ("alignment", sib_ck["alignment"], orig_ck["alignment"]),
                ("r_var", sib_ck["r_var"], orig_ck["r_var"]),
                ("prediction_change.mean_abs_logit_change",
                 sib_ck["prediction_change"]["mean_abs_logit_change"],
                 orig_ck["prediction_change"]["mean_abs_logit_change"]),
                ("prediction_change.rmse_logit_change",
                 sib_ck["prediction_change"]["rmse_logit_change"],
                 orig_ck["prediction_change"]["rmse_logit_change"]),
                ("prediction_change_control.true_effect",
                 sib_ck["prediction_change_control"]["true_effect"],
                 orig_ck["prediction_change_control"]["true_effect"]),
            ):
                rel = abs(a - b) / max(abs(b), 1e-300)
                residuals.append(dict(estimator=estimator, fold_id=sib_fold["fold_id"], field=field,
                                       sib=a, orig=b, rel_residual=rel))

            if sib_fold["fold_id"] != orig_fold["fold_id"]:
                discrete_mismatches.append(f"{estimator}: fold_id {sib_fold['fold_id']} != {orig_fold['fold_id']}")
            if sib_fold["chosen"] != orig_fold["chosen"]:
                discrete_mismatches.append(f"{estimator}/fold{sib_fold['fold_id']}: chosen rank differs")
            sib_exceeds = sib_ck["prediction_change_control"]["exceeds_random"]
            orig_exceeds = orig_ck["prediction_change_control"]["exceeds_random"]
            if sib_exceeds != orig_exceeds:
                discrete_mismatches.append(f"{estimator}/fold{sib_fold['fold_id']}: exceeds_random differs")

    print(json.dumps(dict(
        env_vars=env_vars, blas_threads=blas_threads, residuals=residuals,
        discrete_mismatches=discrete_mismatches,
    )))


if __name__ == "__main__":
    main()
