"""Reliance-measurement battery (Roadmap v3 Step 3).

Pure library -- no CLI/training/eval wiring yet. See:
  subspaces.py    -- class-conditional LDA + cross-fitted-probe subspace estimators
  crossfit.py     -- nested selection/effect cross-fitting harness (grouped folds)
  metrics.py      -- alignment, R_var, projection removal, LEACE, INLP, CMI stub, controls
  uncertainty.py  -- grouped-bootstrap CIs, rank-sensitivity curves

docs/review/AudioShield_Roadmap_v3_AUTHORITATIVE.md, Step 3, is the governing spec.
"""
from __future__ import annotations

from .subspaces import lda_subspace, crossfitted_probe_subspace
from .crossfit import Fold, make_nested_folds, assert_no_group_leakage, derive_groups, run_nested_crossfit
from .metrics import (
    alignment,
    r_var,
    r_var_class_conditional,
    project_out,
    prediction_change,
    LinearEraser,
    fit_leace,
    fit_inlp,
    conditional_mutual_information,
    random_subspace,
    task_direction_subspace,
    removal_control_report,
)
from .uncertainty import grouped_bootstrap_ci, rank_sensitivity_curve

__all__ = [
    "lda_subspace", "crossfitted_probe_subspace",
    "Fold", "make_nested_folds", "assert_no_group_leakage", "derive_groups", "run_nested_crossfit",
    "alignment", "r_var", "r_var_class_conditional", "project_out", "prediction_change",
    "LinearEraser", "fit_leace", "fit_inlp", "conditional_mutual_information",
    "random_subspace", "task_direction_subspace", "removal_control_report",
    "grouped_bootstrap_ci", "rank_sensitivity_curve",
]
