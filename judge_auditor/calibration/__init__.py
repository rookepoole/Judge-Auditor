"""Calibration harness for the Judge Auditor.

Human-label ingestion (labels), judge-vs-human agreement metrics (metrics),
pre-registered protocols (protocols), the calibration join with the
judge/labeler separation rule (calibrate), and blinded re-scoring (rescore).

See SPEC.md in this directory for the frozen calibration rules.
"""

from .adjudicate import (
    ADJUDICATION_RULE,
    adjudicate_all,
    adjudicate_item,
    calibrate_multi,
    calibrate_multi_files,
    fleiss_kappa,
    index_labels,
    labeler_vs_gold,
    pairwise_cohen,
    position_screen,
)
from .calibrate import CalibrationRefusal, calibrate, calibrate_files
from .labels import LABEL_SCHEMA_VERSION, load_labels
from .metrics import calibrate_probe, clopper_pearson_ci
from .protocols import ProtocolMismatch, freeze, verify
from .rescore import compare_rescores, make_rescore_set

__all__ = [
    "ADJUDICATION_RULE",
    "adjudicate_all",
    "adjudicate_item",
    "CalibrationRefusal",
    "calibrate",
    "calibrate_files",
    "calibrate_multi",
    "calibrate_multi_files",
    "fleiss_kappa",
    "index_labels",
    "labeler_vs_gold",
    "LABEL_SCHEMA_VERSION",
    "load_labels",
    "calibrate_probe",
    "clopper_pearson_ci",
    "pairwise_cohen",
    "position_screen",
    "ProtocolMismatch",
    "freeze",
    "verify",
    "compare_rescores",
    "make_rescore_set",
]
