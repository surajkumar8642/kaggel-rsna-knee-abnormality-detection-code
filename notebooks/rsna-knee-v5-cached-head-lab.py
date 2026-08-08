"""Source-only mirror of private Kaggle Version 3 (scriptVersionId 340945234).

Author and test behavior remain browser-first; this file is a reviewable Git
mirror. It contains no outputs, identifiers, reports, predictions,
checkpoints, or DICOM data.
"""

KAGGLE_NOTEBOOK = "surajkumar8642/rsna-knee-v5-cached-head-lab"
KAGGLE_SCRIPT_VERSION_ID = 340945234

# %%

# ============================================================
# Version 5 Task 2 GREEN — immutable run contracts
# ============================================================

from dataclasses import dataclass
from types import MappingProxyType

TARGETS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
)
SOURCE_SCRIPT_VERSION = 340880172
CACHE_SHAPE = (4407, 24, 384)
FOLD_SEED = 20260808
RUN_STAGE = "contract"
ALLOWED_RUN_STAGES = frozenset({"contract", "repro", "screen", "confirm"})


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model_kind: str
    supervision_policy: str
    gold_multiplier: float
    seeds: tuple[int, ...]
    epochs: int = 20
    patience: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3


def _experiment(
    name: str,
    *,
    model_kind: str = "v3_attention",
    supervision_policy: str = "fold_gated",
    gold_multiplier: float = 1.0,
    seeds: tuple[int, ...] = (FOLD_SEED,),
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        model_kind=model_kind,
        supervision_policy=supervision_policy,
        gold_multiplier=gold_multiplier,
        seeds=seeds,
    )


_EXPERIMENTS = {
    "v3_repro": _experiment(
        "v3_repro",
        supervision_policy="raw_v3",
    ),
    "fold_gated": _experiment("fold_gated"),
    "gold_weight_2": _experiment("gold_weight_2", gold_multiplier=2.0),
    "gold_weight_3": _experiment("gold_weight_3", gold_multiplier=3.0),
    "seed_ensemble_3": _experiment(
        "seed_ensemble_3",
        seeds=(FOLD_SEED, FOLD_SEED + 1000, FOLD_SEED + 2000),
    ),
    "bag_mask_tta": _experiment("bag_mask_tta"),
    "ema_head": _experiment("ema_head"),
    "residual_statistics": _experiment(
        "residual_statistics",
        model_kind="residual_statistics",
    ),
    "hierarchical_plane": _experiment(
        "hierarchical_plane",
        model_kind="hierarchical_plane",
    ),
    "second_cv_repeat": _experiment("second_cv_repeat"),
}
EXPERIMENTS = MappingProxyType(_EXPERIMENTS)


def get_experiment(name: str) -> ExperimentConfig:
    try:
        return EXPERIMENTS[name]
    except KeyError as exc:
        raise KeyError(f"unregistered experiment: {name}") from exc


assert RUN_STAGE in ALLOWED_RUN_STAGES
assert tuple(EXPERIMENTS) == (
    "v3_repro",
    "fold_gated",
    "gold_weight_2",
    "gold_weight_3",
    "seed_ensemble_3",
    "bag_mask_tta",
    "ema_head",
    "residual_statistics",
    "hierarchical_plane",
    "second_cv_repeat",
)


    

# %%

# ============================================================
# Version 5 Task 2 regression — immutable run contracts
# ============================================================

from dataclasses import FrozenInstanceError

EXPECTED_TARGETS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
)
EXPECTED_EXPERIMENTS = (
    "v3_repro",
    "fold_gated",
    "gold_weight_2",
    "gold_weight_3",
    "seed_ensemble_3",
    "bag_mask_tta",
    "ema_head",
    "residual_statistics",
    "hierarchical_plane",
    "second_cv_repeat",
)

assert TARGETS == EXPECTED_TARGETS
assert SOURCE_SCRIPT_VERSION == 340880172
assert CACHE_SHAPE == (4407, 24, 384)
assert RUN_STAGE in {"contract", "repro", "screen", "confirm"}
assert tuple(EXPERIMENTS) == EXPECTED_EXPERIMENTS

config = get_experiment("v3_repro")
assert config.name == "v3_repro"
assert config.supervision_policy == "raw_v3"

try:
    config.epochs = 1
except FrozenInstanceError:
    pass
else:
    raise AssertionError("experiment config is mutable")

try:
    EXPERIMENTS["ad_hoc"] = config
except TypeError:
    pass
else:
    raise AssertionError("experiment registry is mutable")

try:
    get_experiment("ad_hoc")
except KeyError as exc:
    assert "unregistered experiment" in str(exc)
else:
    raise AssertionError("ad hoc experiment was not rejected")

print(
    "V5 CONTRACT OBJECTS PASSED: "
    f"targets={len(TARGETS)}, experiments={len(EXPERIMENTS)}, "
    f"source_version={SOURCE_SCRIPT_VERSION}"
)


    

# %%

# ============================================================
# Version 5 Task 3 GREEN — exact sample ordering for all payloads
# ============================================================
def _validated_ids(frame, *, id_column, frame_name):
    if id_column not in frame.columns:
        raise ValueError(f"{frame_name} is missing the ID column")
    raw_ids = frame[id_column]
    missing = raw_ids.isna()
    if bool(missing.any()):
        raise ValueError(f"{frame_name} contains {int(missing.sum())} missing IDs")
    ids = raw_ids.astype("string")
    whitespace_changed = ids.ne(ids.str.strip())
    if bool(whitespace_changed.any()):
        raise ValueError(f"{frame_name} IDs contain leading/trailing whitespace")
    empty = ids.eq("")
    if bool(empty.any()):
        raise ValueError(f"{frame_name} contains {int(empty.sum())} missing IDs")
    duplicate_count = int(ids.duplicated(keep=False).sum())
    if duplicate_count:
        raise ValueError(f"{frame_name} contains {duplicate_count} duplicate ID rows")
    return ids


def _validated_permutation(permutation, row_count):
    raw = np.asarray(permutation)
    if (
        raw.ndim != 1
        or raw.shape[0] != row_count
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        raise ValueError("permutation must be a length-N integer vector")
    result = raw.astype(np.int64, copy=True)
    if (
        np.any(result < 0)
        or np.any(result >= row_count)
        or len(np.unique(result)) != row_count
    ):
        raise ValueError("permutation must contain each row position exactly once")
    return result


def sample_order_permutation(
    test_frame,
    sample_frame,
    *,
    id_column="StudyInstanceUID",
):
    test_ids = _validated_ids(
        test_frame,
        id_column=id_column,
        frame_name="test metadata",
    )
    sample_ids = _validated_ids(
        sample_frame,
        id_column=id_column,
        frame_name="sample submission",
    )
    if len(test_ids) != len(sample_ids) or set(test_ids) != set(sample_ids):
        missing_count = int((~sample_ids.isin(test_ids)).sum())
        extra_count = int((~test_ids.isin(sample_ids)).sum())
        raise ValueError(
            "test/sample ID sets differ "
            f"(missing={missing_count}, extra={extra_count})"
        )
    positions = pd.Series(
        np.arange(len(test_frame), dtype=np.int64),
        index=test_ids.to_numpy(),
    )
    permutation = _validated_permutation(
        positions.loc[sample_ids.to_numpy()].to_numpy(),
        len(test_frame),
    )
    reordered_ids = test_ids.iloc[permutation].tolist()
    if reordered_ids != sample_ids.tolist():
        raise ValueError("permutation does not exactly reproduce sample ID order")
    return permutation


def align_test_payloads_to_submission(
    test_frame,
    sample_frame,
    row_aligned_payloads,
    *,
    id_column="StudyInstanceUID",
):
    permutation = sample_order_permutation(
        test_frame,
        sample_frame,
        id_column=id_column,
    )
    aligned_test = test_frame.iloc[permutation].reset_index(drop=True).copy()
    aligned_test[id_column] = sample_frame[id_column].to_numpy(copy=True)
    if (
        _validated_ids(
            aligned_test,
            id_column=id_column,
            frame_name="aligned test metadata",
        ).tolist()
        != _validated_ids(
            sample_frame,
            id_column=id_column,
            frame_name="sample submission",
        ).tolist()
    ):
        raise AssertionError("internal error: aligned test ID order drifted")

    aligned_payloads = {}
    for payload_name, payload in row_aligned_payloads.items():
        payload_array = np.asarray(payload)
        if payload_array.ndim < 1 or payload_array.shape[0] != len(test_frame):
            raise ValueError(
                f"payload row count mismatch for {payload_name!r}"
            )
        aligned = np.take(payload_array, permutation, axis=0)
        if aligned.shape[0] != len(sample_frame):
            raise AssertionError("internal error: aligned payload row count drifted")
        aligned_payloads[payload_name] = aligned
    return aligned_test, aligned_payloads, permutation


def align_test_features_to_submission(
    test_frame,
    sample_frame,
    features,
    *,
    id_column="StudyInstanceUID",
):
    aligned_test, aligned_payloads, _ = align_test_payloads_to_submission(
        test_frame,
        sample_frame,
        {"features": features},
        id_column=id_column,
    )
    return aligned_test, aligned_payloads["features"]


def build_submission(
    sample_frame,
    predictions,
    *,
    target_columns=TARGETS,
    id_column="StudyInstanceUID",
):
    expected_columns = (id_column, *tuple(target_columns))
    if tuple(sample_frame.columns) != expected_columns:
        raise ValueError("sample submission columns do not match the exact target schema")
    _validated_ids(
        sample_frame,
        id_column=id_column,
        frame_name="sample submission",
    )

    prediction_array = np.asarray(predictions, dtype=np.float64)
    expected_shape = (len(sample_frame), len(target_columns))
    if prediction_array.shape != expected_shape:
        raise ValueError(
            "prediction shape does not match the sample submission "
            f"({prediction_array.shape} != {expected_shape})"
        )
    if not np.isfinite(prediction_array).all():
        raise ValueError("predictions contain non-finite values")
    if bool(((prediction_array < 0.0) | (prediction_array > 1.0)).any()):
        raise ValueError("predictions must be probabilities in [0, 1]")

    submission = sample_frame.loc[:, [id_column]].copy()
    for column_index, target in enumerate(target_columns):
        submission[target] = prediction_array[:, column_index]
    if tuple(submission.columns) != expected_columns:
        raise AssertionError("internal error: output column order drifted")
    if tuple(submission[id_column]) != tuple(sample_frame[id_column]):
        raise AssertionError("internal error: output ID order drifted")
    return submission


    

# %%

# ============================================================
# Version 5 Task 3 TEST — exact IDs, permutation, all row payloads
# ============================================================
import numpy as np
import pandas as pd

ID_COLUMN = "StudyInstanceUID"


def _expect_value_error(callable_object, message_fragment):
    try:
        callable_object()
    except ValueError as exc:
        assert message_fragment in str(exc), str(exc)
    else:
        raise AssertionError(f"expected ValueError containing: {message_fragment}")


_synthetic_test = pd.DataFrame({
    ID_COLUMN: ["study-a", "study-b", "study-c"],
    "marker": [11, 22, 33],
})
_synthetic_sample = pd.DataFrame({
    ID_COLUMN: ["study-c", "study-a", "study-b"],
    **{target: np.zeros(3, dtype=np.float32) for target in TARGETS},
})
_synthetic_payloads = {
    "features": np.array([[101.0, 1.0], [202.0, 2.0], [303.0, 3.0]], dtype=np.float32),
    "planes": np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int8),
    "masks": np.array([[True, False], [True, True], [False, True]]),
    "valid_rows": np.array([True, False, True]),
}

_aligned_test, _aligned_payloads, _permutation = (
    align_test_payloads_to_submission(
        _synthetic_test,
        _synthetic_sample,
        _synthetic_payloads,
        id_column=ID_COLUMN,
    )
)
np.testing.assert_array_equal(_permutation, np.array([2, 0, 1], dtype=np.int64))
assert tuple(_aligned_test[ID_COLUMN]) == tuple(_synthetic_sample[ID_COLUMN])
assert _aligned_test["marker"].tolist() == [33, 11, 22]
np.testing.assert_array_equal(
    _aligned_payloads["features"][:, 0],
    np.array([303.0, 101.0, 202.0], dtype=np.float32),
)
np.testing.assert_array_equal(
    _aligned_payloads["planes"],
    _synthetic_payloads["planes"][_permutation],
)
np.testing.assert_array_equal(
    _aligned_payloads["masks"],
    _synthetic_payloads["masks"][_permutation],
)
np.testing.assert_array_equal(
    _aligned_payloads["valid_rows"],
    _synthetic_payloads["valid_rows"][_permutation],
)

_wrapper_test, _wrapper_features = align_test_features_to_submission(
    _synthetic_test,
    _synthetic_sample,
    _synthetic_payloads["features"],
    id_column=ID_COLUMN,
)
assert tuple(_wrapper_test[ID_COLUMN]) == tuple(_synthetic_sample[ID_COLUMN])
np.testing.assert_array_equal(
    _wrapper_features,
    _synthetic_payloads["features"][_permutation],
)

_synthetic_predictions = np.full(
    (len(_synthetic_sample), len(TARGETS)),
    0.25,
    dtype=np.float32,
)
_synthetic_submission = build_submission(
    _synthetic_sample,
    _synthetic_predictions,
    target_columns=TARGETS,
    id_column=ID_COLUMN,
)
assert tuple(_synthetic_submission.columns) == (ID_COLUMN, *TARGETS)
assert tuple(_synthetic_submission[ID_COLUMN]) == tuple(_synthetic_sample[ID_COLUMN])
assert _synthetic_submission.shape == (3, 13)
assert np.isfinite(_synthetic_submission[list(TARGETS)].to_numpy()).all()
assert (_synthetic_submission[list(TARGETS)].to_numpy() == 0.25).all()

_null_test = _synthetic_test.copy()
_null_test.loc[0, ID_COLUMN] = None
_expect_value_error(
    lambda: align_test_features_to_submission(
        _null_test, _synthetic_sample, _synthetic_payloads["features"]
    ),
    "missing IDs",
)
_whitespace_test = _synthetic_test.copy()
_whitespace_test.loc[0, ID_COLUMN] = "study-a "
_expect_value_error(
    lambda: align_test_features_to_submission(
        _whitespace_test, _synthetic_sample, _synthetic_payloads["features"]
    ),
    "leading/trailing whitespace",
)
_duplicate_test = _synthetic_test.copy()
_duplicate_test.loc[2, ID_COLUMN] = "study-b"
_expect_value_error(
    lambda: align_test_features_to_submission(
        _duplicate_test, _synthetic_sample, _synthetic_payloads["features"]
    ),
    "duplicate ID rows",
)
_duplicate_sample = _synthetic_sample.copy()
_duplicate_sample.loc[2, ID_COLUMN] = "study-a"
_expect_value_error(
    lambda: align_test_features_to_submission(
        _synthetic_test, _duplicate_sample, _synthetic_payloads["features"]
    ),
    "duplicate ID rows",
)
_mismatched_sample = _synthetic_sample.copy()
_mismatched_sample.loc[2, ID_COLUMN] = "study-z"
_expect_value_error(
    lambda: align_test_features_to_submission(
        _synthetic_test, _mismatched_sample, _synthetic_payloads["features"]
    ),
    "ID sets differ",
)
_bad_payloads = dict(_synthetic_payloads)
_bad_payloads["masks"] = _synthetic_payloads["masks"][:2]
_expect_value_error(
    lambda: align_test_payloads_to_submission(
        _synthetic_test, _synthetic_sample, _bad_payloads
    ),
    "payload row count",
)
for _bad_permutation in (
    np.array([0, 0, 2]),
    np.array([0, 1, 3]),
    np.array([0, 1]),
    np.array([0.0, 1.0, 2.0]),
):
    _expect_value_error(
        lambda value=_bad_permutation: _validated_permutation(value, 3),
        "permutation",
    )

_wrong_column_order = _synthetic_sample.loc[
    :, [ID_COLUMN, TARGETS[1], TARGETS[0], *TARGETS[2:]]
]
_expect_value_error(
    lambda: build_submission(_wrong_column_order, _synthetic_predictions),
    "exact target schema",
)
_expect_value_error(
    lambda: build_submission(
        _synthetic_sample, _synthetic_predictions[:, :-1]
    ),
    "prediction shape",
)
_non_finite_predictions = _synthetic_predictions.copy()
_non_finite_predictions[0, 0] = np.nan
_expect_value_error(
    lambda: build_submission(_synthetic_sample, _non_finite_predictions),
    "non-finite",
)
_out_of_range_predictions = _synthetic_predictions.copy()
_out_of_range_predictions[0, 0] = 1.01
_expect_value_error(
    lambda: build_submission(_synthetic_sample, _out_of_range_predictions),
    "probabilities in [0, 1]",
)

print(
    "V5 ORDERING CONTRACT PASSED: exact IDs, verified permutation, "
    "all row payloads, schema, finite/range guards"
)


    

# %%

# ============================================================
# Version 5 Task 4 GREEN — pure fold-local supervision builder
# ============================================================
@dataclass(frozen=True)
class SupervisionBundle:
    values: np.ndarray
    weights: np.ndarray
    enabled_weak_targets: np.ndarray
    policy: str
    gold_multiplier: float


def _validated_row_indices(rows, row_count, *, name):
    raw = np.asarray(rows)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    result = raw.astype(np.int64, copy=True)
    if (
        np.any(result < 0)
        or np.any(result >= row_count)
        or len(np.unique(result)) != len(result)
    ):
        raise ValueError(f"{name} contains invalid or duplicate row indices")
    return result


def build_fold_supervision(
    raw_weak_values,
    raw_weak_weights,
    official_values,
    structural_enabled,
    training_gold_rows,
    heldout_gold_rows,
    *,
    policy,
    gold_multiplier,
    minimum_agreement=0.60,
):
    weak_values_array = np.asarray(raw_weak_values, dtype=np.float32)
    weak_weights_array = np.asarray(raw_weak_weights, dtype=np.float32)
    official_array = np.asarray(official_values, dtype=np.float32)
    structural_array = np.asarray(structural_enabled, dtype=bool)

    if weak_values_array.ndim != 2:
        raise ValueError("weak supervision must be a two-dimensional matrix")
    if weak_values_array.shape != weak_weights_array.shape:
        raise ValueError("weak value/weight shapes differ")
    if official_array.shape != weak_values_array.shape:
        raise ValueError("official and weak supervision shapes differ")
    if structural_array.shape != (weak_values_array.shape[1],):
        raise ValueError("structural gate must contain one flag per target")
    if policy not in {"raw_v3", "fold_gated"}:
        raise ValueError("unsupported supervision policy")
    if not np.isfinite(gold_multiplier) or gold_multiplier <= 0:
        raise ValueError("gold_multiplier must be finite and positive")
    if (
        not np.isfinite(minimum_agreement)
        or minimum_agreement < 0.0
        or minimum_agreement > 1.0
    ):
        raise ValueError("minimum_agreement must be in [0, 1]")

    row_count = weak_values_array.shape[0]
    train_rows = _validated_row_indices(
        training_gold_rows,
        row_count,
        name="training gold rows",
    )
    heldout_rows = _validated_row_indices(
        heldout_gold_rows,
        row_count,
        name="held-out gold rows",
    )
    if np.intersect1d(train_rows, heldout_rows).size:
        raise ValueError("training and held-out gold rows overlap")

    if not np.isfinite(weak_weights_array).all():
        raise ValueError("weak weights must be finite")
    if bool(
        ((weak_weights_array < 0.0) | (weak_weights_array > 1.0)).any()
    ):
        raise ValueError("weak weights must be in [0, 1]")
    positive_weak_weight = weak_weights_array > 0.0
    if bool((positive_weak_weight & ~np.isfinite(weak_values_array)).any()):
        raise ValueError("positive weak weights require finite weak values")
    finite_weak_values = np.isfinite(weak_values_array)
    if bool(
        (
            finite_weak_values
            & ((weak_values_array < 0.0) | (weak_values_array > 1.0))
        ).any()
    ):
        raise ValueError("known weak values must be in [0, 1]")

    if np.isinf(official_array).any():
        raise ValueError("official labels cannot be infinite")
    known_official = np.isfinite(official_array)
    if not np.isin(official_array[known_official], [0.0, 1.0]).all():
        raise ValueError("official labels must be exact binary values")
    all_gold_rows = np.flatnonzero(known_official.any(axis=1))
    supplied_gold_rows = np.union1d(train_rows, heldout_rows)
    if not np.array_equal(all_gold_rows, supplied_gold_rows):
        raise ValueError("fold does not partition every gold row exactly once")

    valid_weak = positive_weak_weight & finite_weak_values
    values = np.where(valid_weak, weak_values_array, np.nan).astype(
        np.float32,
        copy=True,
    )
    weights = np.where(valid_weak, weak_weights_array, 0.0).astype(
        np.float32,
        copy=True,
    )

    if policy == "raw_v3":
        enabled = np.ones(weak_values_array.shape[1], dtype=bool)
    else:
        enabled = structural_array.copy()
        for target_index in range(weak_values_array.shape[1]):
            train_gold_values = official_array[train_rows, target_index]
            train_weak_values = weak_values_array[train_rows, target_index]
            train_weak_weights = weak_weights_array[train_rows, target_index]
            compared = (
                np.isfinite(train_gold_values)
                & np.isfinite(train_weak_values)
                & (train_weak_weights > 0.0)
            )
            if bool(compared.any()):
                parser_binary = train_weak_values[compared] >= 0.5
                gold_binary = train_gold_values[compared] >= 0.5
                agreement = float(np.mean(parser_binary == gold_binary))
                enabled[target_index] &= agreement >= minimum_agreement
        values[:, ~enabled] = np.nan
        weights[:, ~enabled] = 0.0

    training_row_mask = np.zeros(row_count, dtype=bool)
    training_row_mask[train_rows] = True
    training_official = training_row_mask[:, None] & known_official
    values[training_official] = official_array[training_official]
    weights[training_official] = np.float32(gold_multiplier)

    values[heldout_rows, :] = np.nan
    weights[heldout_rows, :] = 0.0
    values[weights == 0.0] = np.nan

    if not np.isfinite(values[weights > 0.0]).all():
        raise AssertionError("internal error: weighted supervision is non-finite")
    if not np.isnan(values[weights == 0.0]).all():
        raise AssertionError("internal error: zero-weight supervision is not erased")
    if bool((weights < 0.0).any()):
        raise AssertionError("internal error: negative supervision weight")

    values.setflags(write=False)
    weights.setflags(write=False)
    enabled.setflags(write=False)
    return SupervisionBundle(
        values=values,
        weights=weights,
        enabled_weak_targets=enabled,
        policy=policy,
        gold_multiplier=float(gold_multiplier),
    )


    

# %%

# ============================================================
# Version 5 Task 4 RED — fold-local supervision and leakage gates
# ============================================================
from dataclasses import FrozenInstanceError
import numpy as np

_synthetic_official = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [np.nan, np.nan, np.nan],
    [np.nan, np.nan, np.nan],
], dtype=np.float32)
_synthetic_weak_values = np.array([
    [0.05, 0.05, 0.05],
    [0.95, 0.95, 0.95],
    [0.95, 0.05, 0.05],
    [0.05, 0.95, 0.95],
    [0.95, 0.80, 0.95],
    [0.05, 0.20, 0.05],
], dtype=np.float32)
_synthetic_weak_weights = np.full((6, 3), 0.50, dtype=np.float32)
_synthetic_structural = np.array([True, True, False])
_synthetic_train_gold = np.array([0, 1], dtype=np.int64)
_synthetic_heldout_gold = np.array([2, 3], dtype=np.int64)
_synthetic_unlabeled = np.array([4, 5], dtype=np.int64)

_weak_values_before = _synthetic_weak_values.copy()
_weak_weights_before = _synthetic_weak_weights.copy()
_official_before = _synthetic_official.copy()

_raw_bundle = build_fold_supervision(
    _synthetic_weak_values,
    _synthetic_weak_weights,
    _synthetic_official,
    _synthetic_structural,
    _synthetic_train_gold,
    _synthetic_heldout_gold,
    policy="raw_v3",
    gold_multiplier=1.0,
)
_gated_bundle = build_fold_supervision(
    _synthetic_weak_values,
    _synthetic_weak_weights,
    _synthetic_official,
    _synthetic_structural,
    _synthetic_train_gold,
    _synthetic_heldout_gold,
    policy="fold_gated",
    gold_multiplier=1.0,
)
_gold2_bundle = build_fold_supervision(
    _synthetic_weak_values,
    _synthetic_weak_weights,
    _synthetic_official,
    _synthetic_structural,
    _synthetic_train_gold,
    _synthetic_heldout_gold,
    policy="fold_gated",
    gold_multiplier=2.0,
)

assert np.array_equal(_synthetic_weak_values, _weak_values_before, equal_nan=True)
assert np.array_equal(_synthetic_weak_weights, _weak_weights_before, equal_nan=True)
assert np.array_equal(_synthetic_official, _official_before, equal_nan=True)
np.testing.assert_array_equal(
    _gated_bundle.enabled_weak_targets,
    np.array([True, False, False]),
)
np.testing.assert_array_equal(
    _raw_bundle.enabled_weak_targets,
    np.array([True, True, True]),
)

for _bundle in (_raw_bundle, _gated_bundle, _gold2_bundle):
    assert np.isnan(_bundle.values[_synthetic_heldout_gold]).all()
    assert np.all(_bundle.weights[_synthetic_heldout_gold] == 0.0)
    assert not _bundle.values.flags.writeable
    assert not _bundle.weights.flags.writeable
    assert not _bundle.enabled_weak_targets.flags.writeable

assert np.isfinite(
    _raw_bundle.values[np.ix_(_synthetic_unlabeled, [1, 2])]
).all()
assert np.isnan(
    _gated_bundle.values[np.ix_(_synthetic_unlabeled, [1, 2])]
).all()
assert np.all(
    _gated_bundle.weights[np.ix_(_synthetic_unlabeled, [1, 2])] == 0.0
)

for _bundle, _multiplier in (
    (_raw_bundle, 1.0),
    (_gated_bundle, 1.0),
    (_gold2_bundle, 2.0),
):
    _known_train = np.isfinite(_synthetic_official[_synthetic_train_gold])
    np.testing.assert_array_equal(
        _bundle.values[_synthetic_train_gold][_known_train],
        _synthetic_official[_synthetic_train_gold][_known_train],
    )
    assert np.all(
        _bundle.weights[_synthetic_train_gold][_known_train] == _multiplier
    )

_gold_values = _synthetic_official[_synthetic_train_gold].reshape(-1)
_gold2_weights = _gold2_bundle.weights[_synthetic_train_gold].reshape(-1)
assert np.any(_gold_values == 0.0) and np.any(_gold_values == 1.0)
assert np.all(_gold2_weights[_gold_values == 0.0] == 2.0)
assert np.all(_gold2_weights[_gold_values == 1.0] == 2.0)
np.testing.assert_array_equal(
    _gated_bundle.weights[_synthetic_unlabeled],
    _gold2_bundle.weights[_synthetic_unlabeled],
)
assert np.array_equal(
    _gated_bundle.values[_synthetic_unlabeled],
    _gold2_bundle.values[_synthetic_unlabeled],
    equal_nan=True,
)

_mutated_weak_values = _synthetic_weak_values.copy()
_mutated_weak_weights = _synthetic_weak_weights.copy()
_mutated_official = _synthetic_official.copy()
_mutated_weak_values[_synthetic_heldout_gold] = 0.95
_mutated_weak_weights[_synthetic_heldout_gold] = 0.70
_mutated_official[_synthetic_heldout_gold] = (
    1.0 - _mutated_official[_synthetic_heldout_gold]
)
_nonheldout = np.array([0, 1, 4, 5], dtype=np.int64)
for _policy in ("raw_v3", "fold_gated"):
    _original = build_fold_supervision(
        _synthetic_weak_values,
        _synthetic_weak_weights,
        _synthetic_official,
        _synthetic_structural,
        _synthetic_train_gold,
        _synthetic_heldout_gold,
        policy=_policy,
        gold_multiplier=2.0,
    )
    _mutated = build_fold_supervision(
        _mutated_weak_values,
        _mutated_weak_weights,
        _mutated_official,
        _synthetic_structural,
        _synthetic_train_gold,
        _synthetic_heldout_gold,
        policy=_policy,
        gold_multiplier=2.0,
    )
    assert np.array_equal(
        _original.values[_nonheldout],
        _mutated.values[_nonheldout],
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        _original.weights[_nonheldout],
        _mutated.weights[_nonheldout],
    )
    np.testing.assert_array_equal(
        _original.enabled_weak_targets,
        _mutated.enabled_weak_targets,
    )

try:
    _gated_bundle.gold_multiplier = 9.0
except (FrozenInstanceError, AttributeError):
    pass
else:
    raise AssertionError("SupervisionBundle is mutable")

def _expect_supervision_error(callable_object, fragment):
    try:
        callable_object()
    except ValueError as exc:
        assert fragment in str(exc), str(exc)
    else:
        raise AssertionError(f"expected ValueError containing: {fragment}")

_expect_supervision_error(
    lambda: build_fold_supervision(
        _synthetic_weak_values,
        _synthetic_weak_weights,
        _synthetic_official,
        _synthetic_structural,
        _synthetic_train_gold,
        _synthetic_heldout_gold,
        policy="unknown",
        gold_multiplier=1.0,
    ),
    "unsupported supervision policy",
)
_expect_supervision_error(
    lambda: build_fold_supervision(
        _synthetic_weak_values,
        _synthetic_weak_weights,
        _synthetic_official,
        _synthetic_structural,
        _synthetic_train_gold,
        _synthetic_heldout_gold,
        policy="fold_gated",
        gold_multiplier=0.0,
    ),
    "gold_multiplier",
)
_expect_supervision_error(
    lambda: build_fold_supervision(
        _synthetic_weak_values,
        _synthetic_weak_weights,
        _synthetic_official,
        _synthetic_structural,
        _synthetic_train_gold,
        np.array([1, 2, 3]),
        policy="fold_gated",
        gold_multiplier=1.0,
    ),
    "overlap",
)

print(
    "V5 SUPERVISION CONTRACT PASSED: raw reproduction isolated; "
    "fold gates use training gold only; heldout erased; gold x2 applies to both classes"
)


    

# %%

# ============================================================
# Version 5 Task 5 GREEN — pinned read-only Version 3 cache
# ============================================================
import hashlib
from pathlib import Path


CACHE_KEYS = (
    "features",
    "planes",
    "masks",
    "decode_failures",
    "missing_planes",
    "fallback_studies",
    "no_series_studies",
    "elapsed_seconds",
)
CACHE_COUNTER_KEYS = (
    "decode_failures",
    "missing_planes",
    "fallback_studies",
    "no_series_studies",
)


@dataclass(frozen=True)
class CacheContract:
    dataset_root: str
    relative_path: str
    expected_sha256: str
    expected_size_bytes: int
    expected_shape: tuple[int, int, int]
    source_script_version: int
    source_run_nonce: str
    slices_per_plane: int
    expected_keys: tuple[str, ...] = CACHE_KEYS


@dataclass(frozen=True)
class FeaturePayload:
    features: np.ndarray
    planes: np.ndarray
    masks: np.ndarray
    counters: object
    elapsed_seconds: float
    cache_sha256: str
    source_script_version: int
    source_run_nonce: str
    slices_per_plane: int
    row_order_evidence: str


TRAIN_CACHE_CONTRACT = CacheContract(
    dataset_root=(
        "/kaggle/input/datasets/surajkumar8642/"
        "rsna-knee-v3-private-cached-features"
    ),
    relative_path=(
        "cache_d59fe0660e284021b6229fbcf5f333aa/"
        "features/train_full_8.npz"
    ),
    expected_sha256=(
        "811a7619e2f5deb556f1268f1c86ee17"
        "ae213910ae446fc996c66d26d549ed9e"
    ),
    expected_size_bytes=75015848,
    expected_shape=(4407, 24, 384),
    source_script_version=340880172,
    source_run_nonce="d59fe0660e284021b6229fbcf5f333aa",
    slices_per_plane=8,
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_exact_cache(contract):
    root = Path(contract.dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError("pinned private cache dataset root is missing")
    expected = (root / contract.relative_path).resolve()
    if not expected.is_relative_to(root):
        raise ValueError("cache relative path escapes the pinned dataset root")
    candidates = sorted(
        path.resolve()
        for path in root.rglob("train_full_8.npz")
        if path.is_file()
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one pinned train cache, found {len(candidates)}"
        )
    if candidates[0] != expected:
        raise ValueError("discovered cache path differs from the pinned relative path")
    if expected.stat().st_size != contract.expected_size_bytes:
        raise ValueError("cache byte size differs from the pinned contract")
    return expected


def validate_feature_arrays(
    arrays,
    *,
    expected_shape,
    slices_per_plane,
    require_zero_fallback=True,
):
    if set(arrays) != set(CACHE_KEYS) or len(arrays) != len(CACHE_KEYS):
        raise ValueError("archive does not contain the exact cache keys")

    features = np.asarray(arrays["features"])
    planes = np.asarray(arrays["planes"])
    masks = np.asarray(arrays["masks"])
    expected_shape = tuple(expected_shape)
    expected_bag_shape = expected_shape[:2]

    if features.shape != expected_shape:
        raise ValueError("feature shape differs from the pinned contract")
    if features.dtype != np.float16:
        raise ValueError("features must use float16")
    if planes.shape != expected_bag_shape or planes.dtype != np.int8:
        raise ValueError("planes must be int8 with the exact bag shape")
    if masks.shape != expected_bag_shape or masks.dtype != np.bool_:
        raise ValueError("masks must be bool with the exact bag shape")
    if not np.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    if not masks.any(axis=1).all():
        raise ValueError("every study must have at least one active instance")
    if bool(np.count_nonzero(features[~masks])):
        raise ValueError("inactive features must be exactly zero")
    if bool(np.count_nonzero(planes[~masks] != -1)):
        raise ValueError("every inactive plane must equal -1")
    if not np.isin(planes[masks], [0, 1, 2]).all():
        raise ValueError("every active plane must be one of 0, 1, 2")
    if bool(np.any(np.all(features[masks] == 0, axis=1))):
        raise ValueError("active feature vectors must not be all zero")
    for plane_index in (0, 1, 2):
        plane_counts = ((planes == plane_index) & masks).sum(axis=1)
        if bool((plane_counts > slices_per_plane).any()):
            raise ValueError("cache exceeds the configured slices per plane")

    counters = {}
    for key in CACHE_COUNTER_KEYS:
        value = np.asarray(arrays[key])
        if value.shape != (1,) or value.dtype != np.int64:
            raise ValueError(f"{key} must be a one-element int64 array")
        counters[key] = int(value[0])
        if counters[key] < 0:
            raise ValueError(f"{key} cannot be negative")
    elapsed = np.asarray(arrays["elapsed_seconds"])
    if (
        elapsed.shape != (1,)
        or elapsed.dtype != np.float64
        or not np.isfinite(elapsed[0])
        or elapsed[0] <= 0
    ):
        raise ValueError("elapsed_seconds must be one positive finite float64")

    plane_present = np.stack(
        [((planes == plane_index) & masks).any(axis=1) for plane_index in (0, 1, 2)],
        axis=1,
    )
    computed_missing_planes = int((~plane_present).sum())
    if counters["missing_planes"] != computed_missing_planes:
        raise ValueError("missing-plane counter disagrees with the plane payload")
    if require_zero_fallback and counters["fallback_studies"] != 0:
        raise ValueError("cache contains fallback studies")
    if counters["no_series_studies"] != 0:
        raise ValueError("cache contains no-series studies")
    return counters, float(elapsed[0])


def load_verified_training_cache(contract):
    if contract.source_script_version != SOURCE_SCRIPT_VERSION:
        raise ValueError("cache source script version differs from the run contract")
    if (
        len(contract.expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in contract.expected_sha256)
    ):
        raise ValueError("cache SHA256 contract is malformed")
    if contract.slices_per_plane <= 0:
        raise ValueError("slices_per_plane must be positive")

    cache_path = _resolve_exact_cache(contract)
    observed_sha256 = _sha256_file(cache_path)
    if observed_sha256 != contract.expected_sha256:
        raise ValueError("cache SHA256 differs from the pinned contract")

    with np.load(cache_path, allow_pickle=False) as archive:
        if tuple(archive.files) != contract.expected_keys:
            raise ValueError("archive key order differs from the pinned contract")
        arrays = {key: archive[key] for key in contract.expected_keys}

    counters, elapsed_seconds = validate_feature_arrays(
        arrays,
        expected_shape=contract.expected_shape,
        slices_per_plane=contract.slices_per_plane,
        require_zero_fallback=True,
    )
    features = arrays["features"]
    planes = arrays["planes"]
    masks = arrays["masks"]
    features.setflags(write=False)
    planes.setflags(write=False)
    masks.setflags(write=False)

    return FeaturePayload(
        features=features,
        planes=planes,
        masks=masks,
        counters=MappingProxyType(counters),
        elapsed_seconds=elapsed_seconds,
        cache_sha256=observed_sha256,
        source_script_version=contract.source_script_version,
        source_run_nonce=contract.source_run_nonce,
        slices_per_plane=contract.slices_per_plane,
        row_order_evidence=(
            "legacy_v3_source_selected_global_train_df_order"
        ),
    )


    

# %%

# ============================================================
# Version 5 Task 5 TEST — exact cache provenance and structure
# ============================================================
from dataclasses import FrozenInstanceError
from pathlib import Path
import numpy as np

_verified_cache = load_verified_training_cache(TRAIN_CACHE_CONTRACT)
assert _verified_cache.source_script_version == 340880172
assert _verified_cache.cache_sha256 == (
    "811a7619e2f5deb556f1268f1c86ee17"
    "ae213910ae446fc996c66d26d549ed9e"
)
assert _verified_cache.features.shape == (4407, 24, 384)
assert _verified_cache.features.dtype == np.float16
assert _verified_cache.planes.shape == (4407, 24)
assert _verified_cache.planes.dtype == np.int8
assert _verified_cache.masks.shape == (4407, 24)
assert _verified_cache.masks.dtype == np.bool_
assert _verified_cache.counters["fallback_studies"] == 0
assert _verified_cache.counters["no_series_studies"] == 0
assert _verified_cache.counters["missing_planes"] == 0
assert _verified_cache.counters["decode_failures"] == 27
assert _verified_cache.slices_per_plane == 8
assert _verified_cache.row_order_evidence == (
    "legacy_v3_source_selected_global_train_df_order"
)
assert not _verified_cache.features.flags.writeable
assert not _verified_cache.planes.flags.writeable
assert not _verified_cache.masks.flags.writeable
assert np.isfinite(_verified_cache.features).all()
assert _verified_cache.masks.all()
assert np.isin(_verified_cache.planes, [0, 1, 2]).all()
for _plane_index in (0, 1, 2):
    assert np.all((_verified_cache.planes == _plane_index).sum(axis=1) == 8)

try:
    TRAIN_CACHE_CONTRACT.expected_sha256 = "mutable"
except (FrozenInstanceError, AttributeError):
    pass
else:
    raise AssertionError("CacheContract is mutable")


def _expect_cache_error(callable_object, fragment):
    try:
        callable_object()
    except (ValueError, FileNotFoundError) as exc:
        assert fragment in str(exc), str(exc)
    else:
        raise AssertionError(f"expected cache error containing: {fragment}")


def _small_cache_arrays():
    features = np.ones((2, 6, 4), dtype=np.float16)
    planes = np.tile(
        np.array([0, 0, 1, 1, 2, -1], dtype=np.int8),
        (2, 1),
    )
    masks = np.tile(
        np.array([True, True, True, True, True, False]),
        (2, 1),
    )
    features[~masks] = 0
    return {
        "features": features,
        "planes": planes,
        "masks": masks,
        "decode_failures": np.array([0], dtype=np.int64),
        "missing_planes": np.array([0], dtype=np.int64),
        "fallback_studies": np.array([0], dtype=np.int64),
        "no_series_studies": np.array([0], dtype=np.int64),
        "elapsed_seconds": np.array([1.0], dtype=np.float64),
    }


def _copy_small_cache():
    return {
        key: value.copy()
        for key, value in _small_cache_arrays().items()
    }


validate_feature_arrays(
    _small_cache_arrays(),
    expected_shape=(2, 6, 4),
    slices_per_plane=2,
)

_missing_key = _copy_small_cache()
_missing_key.pop("planes")
_expect_cache_error(
    lambda: validate_feature_arrays(
        _missing_key,
        expected_shape=(2, 6, 4),
        slices_per_plane=2,
    ),
    "exact cache keys",
)
_wrong_dtype = _copy_small_cache()
_wrong_dtype["features"] = _wrong_dtype["features"].astype(np.float32)
_expect_cache_error(
    lambda: validate_feature_arrays(
        _wrong_dtype,
        expected_shape=(2, 6, 4),
        slices_per_plane=2,
    ),
    "float16",
)
_non_finite = _copy_small_cache()
_non_finite["features"][0, 0, 0] = np.nan
_expect_cache_error(
    lambda: validate_feature_arrays(
        _non_finite,
        expected_shape=(2, 6, 4),
        slices_per_plane=2,
    ),
    "non-finite",
)
_invalid_active_plane = _copy_small_cache()
_invalid_active_plane["planes"][0, 0] = 3
_expect_cache_error(
    lambda: validate_feature_arrays(
        _invalid_active_plane,
        expected_shape=(2, 6, 4),
        slices_per_plane=2,
    ),
    "active plane",
)
_invalid_inactive_plane = _copy_small_cache()
_invalid_inactive_plane["planes"][0, -1] = 0
_expect_cache_error(
    lambda: validate_feature_arrays(
        _invalid_inactive_plane,
        expected_shape=(2, 6, 4),
        slices_per_plane=2,
    ),
    "inactive plane",
)
_invalid_padding = _copy_small_cache()
_invalid_padding["features"][0, -1, 0] = 1
_expect_cache_error(
    lambda: validate_feature_arrays(
        _invalid_padding,
        expected_shape=(2, 6, 4),
        slices_per_plane=2,
    ),
    "inactive features",
)
_empty_study = _copy_small_cache()
_empty_study["masks"][0] = False
_empty_study["planes"][0] = -1
_empty_study["features"][0] = 0
_expect_cache_error(
    lambda: validate_feature_arrays(
        _empty_study,
        expected_shape=(2, 6, 4),
        slices_per_plane=2,
    ),
    "at least one active",
)
_too_many_slices = _copy_small_cache()
_too_many_slices["planes"][0, 2] = 0
_expect_cache_error(
    lambda: validate_feature_arrays(
        _too_many_slices,
        expected_shape=(2, 6, 4),
        slices_per_plane=2,
    ),
    "slices per plane",
)
_fallback_cache = _copy_small_cache()
_fallback_cache["fallback_studies"][0] = 1
_expect_cache_error(
    lambda: validate_feature_arrays(
        _fallback_cache,
        expected_shape=(2, 6, 4),
        slices_per_plane=2,
    ),
    "fallback",
)

print(
    "V5 CACHE CONTRACT PASSED: "
    "source=340880172, rows=4407, bag=24x384, fallbacks=0, exact SHA256"
)


    

# %%

# ============================================================
# Version 5 Task 6 GREEN — immutable CSV-order and fold contract
# ============================================================
@dataclass(frozen=True)
class FoldContract:
    gold_positions: np.ndarray
    gold_values: np.ndarray
    assignment: np.ndarray
    folds: tuple
    fold_sizes: tuple[int, ...]
    n_folds: int
    seed: int
    method: str
    assignment_sha256: str
    study_order_sha256: str
    train_csv_sha256: str
    cache_row_order_evidence: str


def ordered_id_sha256(frame, *, id_column):
    ids = _validated_ids(
        frame,
        id_column=id_column,
        frame_name="training metadata",
    )
    digest = hashlib.sha256()
    for value in ids.tolist():
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def greedy_group_multilabel_folds(labels, groups, n_folds, seed):
    label_array = np.asarray(labels, dtype=np.float64)
    group_array = np.asarray(groups)
    if label_array.ndim != 2 or len(label_array) == 0:
        raise ValueError("labels must be a non-empty two-dimensional matrix")
    if group_array.ndim != 1 or len(group_array) != len(label_array):
        raise ValueError("groups must provide one value per label row")
    if not np.isfinite(label_array).all() or not np.isin(
        label_array, [0.0, 1.0]
    ).all():
        raise ValueError("fold labels must be finite exact binary values")
    if not isinstance(n_folds, (int, np.integer)) or n_folds < 2:
        raise ValueError("n_folds must be an integer of at least two")

    unique_groups, group_codes = np.unique(
        group_array.astype(str),
        return_inverse=True,
    )
    if n_folds > len(unique_groups):
        raise ValueError("n_folds cannot exceed the number of groups")

    group_labels = np.zeros(
        (len(unique_groups), label_array.shape[1]),
        dtype=np.float64,
    )
    group_sizes = np.zeros(len(unique_groups), dtype=np.int64)
    for group_index in range(len(unique_groups)):
        members = group_codes == group_index
        group_labels[group_index] = label_array[members].max(axis=0)
        group_sizes[group_index] = int(members.sum())

    positive_totals = group_labels.sum(axis=0).clip(min=1.0)
    rarity = (group_labels / positive_totals).sum(axis=1)
    rng = np.random.default_rng(seed)
    tie_noise = rng.random(len(unique_groups)) * 1e-9
    order = np.lexsort((tie_noise, -group_sizes, -rarity))

    fold_positive = np.zeros(
        (n_folds, label_array.shape[1]),
        dtype=np.float64,
    )
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    group_fold = np.full(len(unique_groups), -1, dtype=np.int64)
    desired_positive = group_labels.sum(axis=0) / n_folds
    desired_size = len(label_array) / n_folds
    base_capacity = len(label_array) // n_folds
    fold_capacity = np.full(n_folds, base_capacity, dtype=np.int64)
    fold_capacity[: len(label_array) % n_folds] += 1

    for group_index in order:
        costs = []
        for fold_index in range(n_folds):
            if (
                fold_sizes[fold_index] + group_sizes[group_index]
                > fold_capacity[fold_index]
            ):
                continue
            positive_after = (
                fold_positive[fold_index] + group_labels[group_index]
            )
            positive_cost = np.mean(
                (
                    (positive_after - desired_positive)
                    / np.maximum(desired_positive, 1.0)
                )
                ** 2
            )
            size_after = fold_sizes[fold_index] + group_sizes[group_index]
            size_cost = (
                (size_after - desired_size) / max(desired_size, 1.0)
            ) ** 2
            costs.append(
                (
                    positive_cost + 0.20 * size_cost,
                    fold_sizes[fold_index],
                    fold_index,
                )
            )
        if not costs:
            raise ValueError(
                "group sizes cannot satisfy deterministic fold capacities"
            )
        chosen_fold = min(costs)[2]
        group_fold[group_index] = chosen_fold
        fold_positive[chosen_fold] += group_labels[group_index]
        fold_sizes[chosen_fold] += group_sizes[group_index]

    assignment = group_fold[group_codes]
    if np.any(assignment < 0):
        raise AssertionError("internal error: fold assignment is incomplete")
    return assignment.astype(np.int64, copy=False)


def load_training_fold_contract(
    train_csv_path,
    *,
    target_columns,
    id_column,
    n_folds,
    seed,
):
    path = Path(train_csv_path)
    if not path.is_file():
        raise FileNotFoundError("competition train.csv is missing")
    train_frame = pd.read_csv(path)
    expected_columns = (id_column, "Report", *tuple(target_columns))
    if tuple(train_frame.columns) != expected_columns:
        raise ValueError("train.csv does not match the exact ordered schema")
    if len(train_frame) != TRAIN_CACHE_CONTRACT.expected_shape[0]:
        raise ValueError("train.csv row count differs from the pinned cache")

    ids = _validated_ids(
        train_frame,
        id_column=id_column,
        frame_name="train.csv",
    )
    official_values = train_frame[list(target_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    if np.isinf(official_values).any():
        raise ValueError("official labels contain infinite values")
    gold_row_mask = np.isfinite(official_values).any(axis=1)
    partial_gold = gold_row_mask & ~np.isfinite(official_values).all(axis=1)
    if bool(partial_gold.any()):
        raise ValueError("officially labelled rows must contain all targets")
    gold_positions = np.flatnonzero(gold_row_mask).astype(np.int64)
    gold_values = official_values[gold_positions].astype(
        np.float32,
        copy=True,
    )
    if len(gold_positions) != 58:
        raise ValueError("expected exactly 58 officially labelled studies")
    if not np.isin(gold_values, [0.0, 1.0]).all():
        raise ValueError("official labels must be exact binary values")

    groups = ids.iloc[gold_positions].to_numpy()
    assignment = greedy_group_multilabel_folds(
        gold_values,
        groups,
        n_folds,
        seed,
    )
    if not np.all((assignment >= 0) & (assignment < n_folds)):
        raise AssertionError("internal error: fold assignment is out of range")

    fold_pairs = []
    validation_counts = np.zeros(len(gold_positions), dtype=np.int8)
    fold_sizes = []
    for fold_index in range(n_folds):
        training_index = np.flatnonzero(
            assignment != fold_index
        ).astype(np.int64)
        validation_index = np.flatnonzero(
            assignment == fold_index
        ).astype(np.int64)
        if len(np.intersect1d(training_index, validation_index)):
            raise AssertionError("internal error: train/validation overlap")
        validation_counts[validation_index] += 1
        fold_sizes.append(len(validation_index))
        training_index.setflags(write=False)
        validation_index.setflags(write=False)
        fold_pairs.append((training_index, validation_index))
    if not np.all(validation_counts == 1):
        raise AssertionError("internal error: gold OOF coverage is not exact")
    expected_sizes = tuple(
        len(gold_positions) // n_folds
        + (1 if index < len(gold_positions) % n_folds else 0)
        for index in range(n_folds)
    )
    if tuple(fold_sizes) != expected_sizes:
        raise ValueError("fold sizes differ from deterministic capacities")

    assignment_sha256 = hashlib.sha256(
        assignment.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    study_order_sha256 = ordered_id_sha256(
        train_frame,
        id_column=id_column,
    )
    train_csv_sha256 = _sha256_file(path)

    gold_positions.setflags(write=False)
    gold_values.setflags(write=False)
    assignment.setflags(write=False)
    return train_frame, FoldContract(
        gold_positions=gold_positions,
        gold_values=gold_values,
        assignment=assignment,
        folds=tuple(fold_pairs),
        fold_sizes=tuple(fold_sizes),
        n_folds=int(n_folds),
        seed=int(seed),
        method="greedy-multilabel-study",
        assignment_sha256=assignment_sha256,
        study_order_sha256=study_order_sha256,
        train_csv_sha256=train_csv_sha256,
        cache_row_order_evidence=(
            "legacy_v3_source_selected_global_train_df_order"
        ),
    )


    

# %%

# ============================================================
# Version 5 Task 6 RED — train-order and deterministic fold contract
# ============================================================
from dataclasses import FrozenInstanceError
from pathlib import Path
import numpy as np
import pandas as pd

TRAIN_CSV_PATH = Path(
    "/kaggle/input/competitions/"
    "rsna-knee-abnormality-detection/train.csv"
)
TRAIN_DF, FOLD_CONTRACT = load_training_fold_contract(
    TRAIN_CSV_PATH,
    target_columns=TARGETS,
    id_column=ID_COLUMN,
    n_folds=5,
    seed=FOLD_SEED,
)

assert TRAIN_DF.shape == (4407, 14)
assert tuple(TRAIN_DF.columns) == (ID_COLUMN, "Report", *TARGETS)
assert len(FOLD_CONTRACT.gold_positions) == 58
assert FOLD_CONTRACT.gold_values.shape == (58, 12)
assert np.isfinite(FOLD_CONTRACT.gold_values).all()
assert np.isin(FOLD_CONTRACT.gold_values, [0.0, 1.0]).all()
assert FOLD_CONTRACT.fold_sizes == (12, 12, 12, 11, 11)
assert FOLD_CONTRACT.method == "greedy-multilabel-study"
assert FOLD_CONTRACT.n_folds == 5
assert FOLD_CONTRACT.seed == 20260808
assert len(FOLD_CONTRACT.assignment_sha256) == 64
assert len(FOLD_CONTRACT.study_order_sha256) == 64
assert len(FOLD_CONTRACT.train_csv_sha256) == 64
assert FOLD_CONTRACT.assignment_sha256 == (
    "1aed5f3ad0a84ee7d3a7abc2270b60b6"
    "48d6b52ee7c3334f8bb51736b8b9cd01"
)
assert FOLD_CONTRACT.study_order_sha256 == (
    "1196750855320f5e465273649763fc10"
    "691ebc30446936ada20f9c515ab0a48e"
)
assert FOLD_CONTRACT.train_csv_sha256 == (
    "8ca2203c0e9d61c080c7a314c7cdb51"
    "c1b03a1d9eb4770819f7f34af53ef4e33"
)
assert FOLD_CONTRACT.cache_row_order_evidence == (
    "legacy_v3_source_selected_global_train_df_order"
)
assert not FOLD_CONTRACT.gold_positions.flags.writeable
assert not FOLD_CONTRACT.gold_values.flags.writeable
assert not FOLD_CONTRACT.assignment.flags.writeable
assert len(FOLD_CONTRACT.folds) == 5

_validation_counts = np.zeros(58, dtype=np.int8)
for _fold_index, (_training_index, _validation_index) in enumerate(
    FOLD_CONTRACT.folds
):
    assert not _training_index.flags.writeable
    assert not _validation_index.flags.writeable
    assert len(np.intersect1d(_training_index, _validation_index)) == 0
    assert len(_training_index) + len(_validation_index) == 58
    assert np.all(FOLD_CONTRACT.assignment[_validation_index] == _fold_index)
    _validation_counts[_validation_index] += 1
assert np.all(_validation_counts == 1)

try:
    FOLD_CONTRACT.seed = 1
except (FrozenInstanceError, AttributeError):
    pass
else:
    raise AssertionError("FoldContract is mutable")

_synthetic_labels = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 0],
    [1, 1, 1],
    [1, 0, 0],
    [0, 1, 0],
], dtype=np.float32)
_synthetic_groups = np.array(
    [f"group-{index}" for index in range(len(_synthetic_labels))]
)
_synthetic_a = greedy_group_multilabel_folds(
    _synthetic_labels,
    _synthetic_groups,
    n_folds=2,
    seed=20260808,
)
_synthetic_b = greedy_group_multilabel_folds(
    _synthetic_labels,
    _synthetic_groups,
    n_folds=2,
    seed=20260808,
)
np.testing.assert_array_equal(_synthetic_a, _synthetic_b)
assert sorted(np.bincount(_synthetic_a, minlength=2).tolist()) == [5, 5]

print(
    "V5 FOLD CONTRACT PASSED: "
    f"gold=58, sizes={FOLD_CONTRACT.fold_sizes}, "
    f"assignment_sha256={FOLD_CONTRACT.assignment_sha256}, "
    f"train_order_sha256={FOLD_CONTRACT.study_order_sha256}"
)


    

# %%

# ============================================================
# Version 5 Task 7 GREEN — high-precision report supervision
# ============================================================
import re
import unicodedata


@dataclass(frozen=True)
class WeakLabel:
    state: str
    value: float
    weight: float
    conflict: bool = False


@dataclass(frozen=True)
class ReportSupervision:
    weak_values: np.ndarray
    weak_weights: np.ndarray
    weak_conflicts: np.ndarray
    official_values: np.ndarray
    structural_enabled: np.ndarray
    coverage: np.ndarray
    positive_prevalence: np.ndarray
    gold_agreement: np.ndarray
    gold_row_count: int
    unlabeled_row_count: int
    weak_digest: str


UNKNOWN = WeakLabel("unknown", np.nan, 0.0)
CONFIDENT_POSITIVE = WeakLabel("positive", 0.95, 0.70)
PROBABLE_POSITIVE = WeakLabel("positive", 0.80, 0.40)
CONFIDENT_NEGATIVE = WeakLabel("negative", 0.05, 0.50)


TARGET_CONCEPTS = {
    "ACL": (
        r"\bacl\b",
        r"\banterior cruciate ligament\b",
        r"\bligamento cruzado anterior\b",
        r"\blca\b",
    ),
    "MCL": (
        r"\bmcl\b",
        r"\bmedial collateral ligament\b",
        r"\bligamento colateral medial\b",
        r"\blcm\b",
    ),
    "Medial Meniscus": (
        r"\bmedial menisc(?:us|al)\b",
        r"\bmenisc(?:us|al)[ -]medial\b",
        r"\bmenisco medial\b",
        r"\bmenisque medial\b",
    ),
    "Lateral Meniscus": (
        r"\blateral menisc(?:us|al)\b",
        r"\bmenisc(?:us|al)[ -]lateral\b",
        r"\bmenisco lateral\b",
        r"\bmenisque lateral\b",
    ),
    "Medial OA": (
        r"\bmedial (?:compartment )?(?:oa|osteoarth(?:ritis|rosis)|arthrosis)\b",
        r"\b(?:oa|osteoarth(?:ritis|rosis)|arthrosis) (?:of (?:the )?)?medial compartment\b",
        r"\b(?:medial compartment|compartimento medial).{0,60}(?:degenerative change|joint space narrowing|chondrosis|cartilage loss|osteoartritis|artrosis)\b",
        r"\b(?:artrosis|osteoartritis) (?:del )?compartimento medial\b",
    ),
    "Lateral OA": (
        r"\blateral (?:compartment )?(?:oa|osteoarth(?:ritis|rosis)|arthrosis)\b",
        r"\b(?:oa|osteoarth(?:ritis|rosis)|arthrosis) (?:of (?:the )?)?lateral compartment\b",
        r"\b(?:lateral compartment|compartimento lateral).{0,60}(?:degenerative change|joint space narrowing|chondrosis|cartilage loss|osteoartritis|artrosis)\b",
        r"\b(?:artrosis|osteoartritis) (?:del )?compartimento lateral\b",
    ),
    "PF OA": (
        r"\bpatello?-?femoral (?:oa|osteoarth(?:ritis|rosis)|arthrosis|chondrosis|degeneration|cartilage loss)\b",
        r"\b(?:oa|osteoarth(?:ritis|rosis)|arthrosis) (?:of (?:the )?)?patello?-?femoral compartment\b",
        r"\b(?:artrosis|osteoartritis) patelofemoral\b",
    ),
    "Effusion": (
        r"\b(?:joint |articular )?effusion\b",
        r"\bderrame articular\b",
        r"\bgelenkerguss\b",
        r"\bepanchement articulaire\b",
    ),
    "Synovitis": (
        r"\bsynovitis\b",
        r"\bsinovitis\b",
        r"\bsynovial inflammation\b",
    ),
    "Baker's": (
        r"\bbaker'?s?[ -](?:cyst|cystic lesion)\b",
        r"\bpopliteal cyst\b",
        r"\bquiste (?:de )?baker\b",
        r"\bquiste popliteo\b",
        r"\bbaker[ -]zyste\b",
    ),
    "Contusion": (
        r"\b(?:marrow|bone|osseous) contusion\b",
        r"\bbone bruise\b",
        r"\bcontusion (?:medular|osea)\b",
        r"\bedema oseo traumatico\b",
        r"\btraumatic bone marrow edema\b",
    ),
    "Fracture": (
        r"\bfracture\b",
        r"\bfractura\b",
        r"\bfraktur\b",
        r"\b(?:cortical|trabecular) break\b",
    ),
}
COMPILED_TARGET_CONCEPTS = {
    target: re.compile("(?:" + "|".join(patterns) + ")")
    for target, patterns in TARGET_CONCEPTS.items()
}

NEGATION = re.compile(
    r"\b(?:no|not|without|absent|negative for|free of|neither|nor|"
    r"no evidence of|no sign of|sin|ausencia de|negativo para|no hay|"
    r"sem|kein|keine|pas de)\b"
)
UNCERTAINTY = re.compile(
    r"\b(?:cannot exclude|can(?:not|'t) rule out|not excluded|indeterminate|"
    r"equivocal|questionable|possible|possibly|may represent|could represent|"
    r"uncertain|cannot assess|limited evaluation|no se puede excluir|"
    r"indeterminado|dudoso|posible)\b"
)
HISTORY = re.compile(
    r"\b(?:status post|s/?p|history of|historical|previous|previously|prior|"
    r"remote|old|postoperative|postsurgical|repaired|reconstruction|repair|"
    r"antecedente de|estado posterior a|previamente|antigu[oa])\b"
)
PROBABLE = re.compile(
    r"\b(?:probable|probably|likely|most likely|favou?rs?|suspicious for|"
    r"consistent with|compatible with|presumed|presumptive|"
    r"probablemente|compatible con|sugestivo de)\b"
)
EXPLICIT_NEGATIVE = re.compile(
    r"\b(?:intact|preserved|unremarkable|normal|sin desgarro|integro|intacto)\b"
)
POSITIVE_FINDING = re.compile(
    r"\b(?:tear|torn|rupture|ruptured|sprain|injury|complete|partial|"
    r"degenerative|complex|radial|horizontal|flap|bucket-handle|"
    r"osteoarth(?:ritis|rosis)|arthrosis|oa|chondrosis|cartilage loss|"
    r"joint space narrowing|effusion|synovitis|cyst|contusion|bruise|"
    r"fracture|desgarro|rotura|artrosis|osteoartritis|derrame|sinovitis|"
    r"quiste|fractura|fraktur)\b"
)
CURRENT_FINDING = re.compile(
    r"\b(?:acute|new|current|recurrent|re-tear|retear|again|active|"
    r"agud[oa]|nuev[oa]|actual|recurrente)\b"
)
POST_NEGATION = re.compile(
    r"^\W*(?:is |are |appears? )?(?:absent|not seen|not present|intact|"
    r"preserved|unremarkable|normal|ausente|intacto|integro)\b"
)


def normalize_report(text):
    value = "" if pd.isna(text) else str(text)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )
    value = value.casefold().replace("â€™", "'")
    return re.sub(r"\s+", " ", value).strip()


def report_clauses_from_normalized(normalized):
    return [
        clause.strip()
        for clause in re.split(
            r"(?:[.;:\n]+|,\s*(?:but|however|although|pero|sin embargo)\b|"
            r"\b(?:but|however|although|whereas|pero|sin embargo)\b)",
            normalized,
        )
        if clause.strip()
    ]


def trim_left_scope(text):
    parts = re.split(
        r"(?:,|\b(?:and|but|however|whereas|pero)\b)",
        text,
    )
    return parts[-1] if parts else text


def trim_right_scope(text):
    return re.split(
        r"(?:,|\b(?:and|but|however|whereas|pero)\b)",
        text,
        maxsplit=1,
    )[0]


def label_mention(clause, match, radius=72):
    left = trim_left_scope(
        clause[max(0, match.start() - radius):match.start()]
    )
    right = trim_right_scope(
        clause[match.end():match.end() + radius]
    )
    local = f"{left} {match.group(0)} {right}"

    if UNCERTAINTY.search(local):
        return UNKNOWN
    if (
        NEGATION.search(left)
        or EXPLICIT_NEGATIVE.search(left)
        or POST_NEGATION.search(right)
    ):
        return CONFIDENT_NEGATIVE
    if CURRENT_FINDING.search(local) and POSITIVE_FINDING.search(local):
        return CONFIDENT_POSITIVE
    if HISTORY.search(local):
        return UNKNOWN
    if PROBABLE.search(local) and POSITIVE_FINDING.search(local):
        return PROBABLE_POSITIVE
    if POSITIVE_FINDING.search(local):
        return CONFIDENT_POSITIVE
    return UNKNOWN


def aggregate_mentions(labels):
    decisive_states = {
        label.state
        for label in labels
        if label.state != "unknown"
    }
    if len(decisive_states) > 1:
        return WeakLabel("unknown", np.nan, 0.0, True)
    if decisive_states:
        state = next(iter(decisive_states))
        return max(
            (
                label
                for label in labels
                if label.state == state
            ),
            key=lambda label: label.weight,
        )
    return UNKNOWN


def analyze_report(text):
    normalized = normalize_report(text)
    clauses = report_clauses_from_normalized(normalized)
    labels_by_target = {
        target: []
        for target in TARGETS
    }
    for clause in clauses:
        for target, concept in COMPILED_TARGET_CONCEPTS.items():
            labels_by_target[target].extend(
                label_mention(clause, match)
                for match in concept.finditer(clause)
            )
    return {
        target: aggregate_mentions(labels)
        for target, labels in labels_by_target.items()
    }


def classify_target_mention(text, target):
    if target not in COMPILED_TARGET_CONCEPTS:
        raise ValueError("unsupported report target")
    return analyze_report(text)[target]


def _report_supervision_digest(
    weak_values,
    weak_weights,
    weak_conflicts,
    structural_enabled,
):
    digest = hashlib.sha256()
    digest.update(
        np.nan_to_num(
            weak_values,
            nan=-1.0,
        ).astype("<f4", copy=False).tobytes()
    )
    digest.update(
        weak_weights.astype("<f4", copy=False).tobytes()
    )
    digest.update(
        weak_conflicts.astype(np.uint8, copy=False).tobytes()
    )
    digest.update(
        structural_enabled.astype(np.uint8, copy=False).tobytes()
    )
    return digest.hexdigest()


def build_report_supervision(train_frame, *, target_columns):
    if tuple(target_columns) != TARGETS:
        raise ValueError("report target schema differs from the run contract")
    report_candidates = [
        column
        for column in train_frame.columns
        if column.casefold() in {
            "report",
            "reporttext",
            "report_text",
        }
    ]
    if len(report_candidates) != 1:
        raise ValueError("expected exactly one report column")
    report_column = report_candidates[0]

    weak_values = np.full(
        (len(train_frame), len(target_columns)),
        np.nan,
        dtype=np.float32,
    )
    weak_weights = np.zeros_like(weak_values)
    weak_conflicts = np.zeros_like(
        weak_values,
        dtype=bool,
    )
    for row_position, report in enumerate(train_frame[report_column]):
        labels = analyze_report(report)
        for target_position, target in enumerate(target_columns):
            parsed = labels[target]
            weak_values[row_position, target_position] = parsed.value
            weak_weights[row_position, target_position] = parsed.weight
            weak_conflicts[row_position, target_position] = (
                parsed.conflict
            )
    weak_values[weak_weights == 0.0] = np.nan

    official_values = train_frame[list(target_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    if np.isinf(official_values).any():
        raise ValueError("official labels contain infinite values")
    gold_rows = np.isfinite(official_values).any(axis=1)
    if bool(
        (
            gold_rows
            & ~np.isfinite(official_values).all(axis=1)
        ).any()
    ):
        raise ValueError("officially labelled rows are incomplete")
    permanent_unlabeled = ~gold_rows

    structural_enabled = np.zeros(
        len(target_columns),
        dtype=bool,
    )
    coverage = np.zeros(
        len(target_columns),
        dtype=np.float64,
    )
    positive_prevalence = np.full(
        len(target_columns),
        np.nan,
        dtype=np.float64,
    )
    gold_agreement = np.full(
        len(target_columns),
        np.nan,
        dtype=np.float64,
    )

    for target_position in range(len(target_columns)):
        covered_all = weak_weights[:, target_position] > 0
        covered_unlabeled = covered_all & permanent_unlabeled
        coverage[target_position] = float(
            covered_unlabeled.sum() / permanent_unlabeled.sum()
        )
        if bool(covered_unlabeled.any()):
            positive_prevalence[target_position] = float(
                (
                    weak_values[
                        covered_unlabeled,
                        target_position,
                    ]
                    >= 0.5
                ).mean()
            )

        gold_known = np.isfinite(
            official_values[:, target_position]
        )
        gold_covered = gold_known & covered_all
        if bool(gold_covered.any()):
            gold_agreement[target_position] = float(
                (
                    (
                        weak_values[
                            gold_covered,
                            target_position,
                        ]
                        >= 0.5
                    )
                    == (
                        official_values[
                            gold_covered,
                            target_position,
                        ]
                        >= 0.5
                    )
                ).mean()
            )

        structural_enabled[target_position] = (
            0.01 <= coverage[target_position] <= 0.98
            and np.isfinite(
                positive_prevalence[target_position]
            )
            and (
                0.001
                <= positive_prevalence[target_position]
                <= 0.80
            )
        )

    if not np.isfinite(
        weak_values[weak_weights > 0]
    ).all():
        raise AssertionError(
            "internal error: weighted weak label is non-finite"
        )
    if not np.isnan(
        weak_values[weak_weights == 0]
    ).all():
        raise AssertionError(
            "internal error: unknown weak label is not erased"
        )
    if bool(
        ((weak_weights < 0.0) | (weak_weights > 1.0)).any()
    ):
        raise AssertionError(
            "internal error: weak weights are outside [0, 1]"
        )

    weak_digest = _report_supervision_digest(
        weak_values,
        weak_weights,
        weak_conflicts,
        structural_enabled,
    )
    for array in (
        weak_values,
        weak_weights,
        weak_conflicts,
        official_values,
        structural_enabled,
        coverage,
        positive_prevalence,
        gold_agreement,
    ):
        array.setflags(write=False)

    return ReportSupervision(
        weak_values=weak_values,
        weak_weights=weak_weights,
        weak_conflicts=weak_conflicts,
        official_values=official_values,
        structural_enabled=structural_enabled,
        coverage=coverage,
        positive_prevalence=positive_prevalence,
        gold_agreement=gold_agreement,
        gold_row_count=int(gold_rows.sum()),
        unlabeled_row_count=int(permanent_unlabeled.sum()),
        weak_digest=weak_digest,
    )


    

# %%

# ============================================================
# Version 5 Task 7 RED — report parser and actual fold supervision
# ============================================================
from dataclasses import FrozenInstanceError
import hashlib
import numpy as np

_PARSER_CASES = (
    ("complete acl tear", "ACL", "positive"),
    ("no acl tear", "ACL", "negative"),
    ("cannot exclude acl tear", "ACL", "unknown"),
    ("status post acl repair", "ACL", "unknown"),
    ("large joint effusion", "Effusion", "positive"),
    ("no joint effusion", "Effusion", "negative"),
    ("degenerative medial meniscus tear", "Medial Meniscus", "positive"),
    ("no acute fracture", "Fracture", "negative"),
    ("marrow contusion lateral femoral condyle", "Contusion", "positive"),
    ("small baker cyst", "Baker's", "positive"),
    ("no fracture but complete acl tear", "ACL", "positive"),
    ("no fracture but complete acl tear", "Fracture", "negative"),
    ("status post acl reconstruction with recurrent tear", "ACL", "positive"),
    (
        "cannot exclude medial meniscal tear; definite joint effusion",
        "Medial Meniscus",
        "unknown",
    ),
    (
        "cannot exclude medial meniscal tear; definite joint effusion",
        "Effusion",
        "positive",
    ),
    ("no acl or mcl tear", "ACL", "negative"),
    ("no acl or mcl tear", "MCL", "negative"),
    ("intact acl and medial meniscus tear", "ACL", "negative"),
    (
        "intact acl and medial meniscus tear",
        "Medial Meniscus",
        "positive",
    ),
)
for _report_text, _target, _expected_state in _PARSER_CASES:
    _parsed = classify_target_mention(_report_text, _target)
    assert _parsed.state == _expected_state, (
        _target,
        _expected_state,
        _parsed.state,
    )

REPORT_SUPERVISION = build_report_supervision(
    TRAIN_DF,
    target_columns=TARGETS,
)
assert REPORT_SUPERVISION.weak_values.shape == (4407, 12)
assert REPORT_SUPERVISION.weak_weights.shape == (4407, 12)
assert REPORT_SUPERVISION.weak_conflicts.shape == (4407, 12)
assert REPORT_SUPERVISION.official_values.shape == (4407, 12)
assert REPORT_SUPERVISION.structural_enabled.shape == (12,)
assert REPORT_SUPERVISION.weak_digest == (
    "fa067bd6680a4bb2eb3a864a12e90e69"
    "b214e67760a4d316cfc133c9bd55d0f4"
)
assert REPORT_SUPERVISION.gold_row_count == 58
assert REPORT_SUPERVISION.unlabeled_row_count == 4349
assert int(REPORT_SUPERVISION.structural_enabled.sum()) == 8
assert tuple(
    target
    for target, enabled in zip(
        TARGETS,
        REPORT_SUPERVISION.structural_enabled,
    )
    if not enabled
) == ("Medial OA", "Lateral OA", "PF OA", "Synovitis")
assert not REPORT_SUPERVISION.weak_values.flags.writeable
assert not REPORT_SUPERVISION.weak_weights.flags.writeable
assert not REPORT_SUPERVISION.weak_conflicts.flags.writeable
assert not REPORT_SUPERVISION.official_values.flags.writeable
assert not REPORT_SUPERVISION.structural_enabled.flags.writeable
assert np.isfinite(
    REPORT_SUPERVISION.weak_values[
        REPORT_SUPERVISION.weak_weights > 0
    ]
).all()
assert np.isnan(
    REPORT_SUPERVISION.weak_values[
        REPORT_SUPERVISION.weak_weights == 0
    ]
).all()
assert np.all(
    (REPORT_SUPERVISION.weak_weights >= 0.0)
    & (REPORT_SUPERVISION.weak_weights <= 1.0)
)

_fold_enabled_counts = []
for _fold_index, (_gold_train_index, _gold_valid_index) in enumerate(
    FOLD_CONTRACT.folds
):
    _training_gold_rows = FOLD_CONTRACT.gold_positions[
        _gold_train_index
    ]
    _heldout_gold_rows = FOLD_CONTRACT.gold_positions[
        _gold_valid_index
    ]
    _bundle = build_fold_supervision(
        REPORT_SUPERVISION.weak_values,
        REPORT_SUPERVISION.weak_weights,
        REPORT_SUPERVISION.official_values,
        REPORT_SUPERVISION.structural_enabled,
        _training_gold_rows,
        _heldout_gold_rows,
        policy="fold_gated",
        gold_multiplier=1.0,
    )
    assert np.isnan(_bundle.values[_heldout_gold_rows]).all()
    assert np.all(_bundle.weights[_heldout_gold_rows] == 0.0)
    _fold_enabled_counts.append(
        int(_bundle.enabled_weak_targets.sum())
    )

try:
    REPORT_SUPERVISION.gold_row_count = 0
except (FrozenInstanceError, AttributeError):
    pass
else:
    raise AssertionError("ReportSupervision is mutable")

print(
    "V5 REPORT SUPERVISION PASSED: "
    f"parser_cases={len(_PARSER_CASES)}, enabled_global=8, "
    f"fold_enabled_counts={tuple(_fold_enabled_counts)}, "
    f"weak_digest={REPORT_SUPERVISION.weak_digest}"
)


    

# %%

# ============================================================
# Version 5 Task 8 GREEN — cached MIL model, dataset, and loss
# ============================================================
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class TargetAttentionMILV3(nn.Module):
    def __init__(
        self,
        feature_dim=384,
        num_planes=3,
        num_targets=12,
        token_dim=192,
        plane_dim=32,
        hidden_dim=96,
        dropout=0.20,
    ):
        super().__init__()
        if not 0 < plane_dim < token_dim:
            raise ValueError("plane_dim must be smaller than token_dim")
        self.feature_dim = int(feature_dim)
        self.num_planes = int(num_planes)
        self.num_targets = int(num_targets)
        image_dim = token_dim - plane_dim

        self.feature_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, image_dim),
            nn.GELU(),
        )
        self.plane_embedding = nn.Embedding(
            num_planes,
            plane_dim,
        )
        self.token_fusion = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim),
        )
        self.target_queries = nn.Parameter(
            torch.empty(num_targets, token_dim)
        )
        nn.init.trunc_normal_(self.target_queries, std=0.02)
        self.attention_scale = token_dim ** -0.5
        self.head_trunk = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_weight = nn.Parameter(
            torch.empty(num_targets, hidden_dim)
        )
        self.output_bias = nn.Parameter(
            torch.zeros(num_targets)
        )
        nn.init.xavier_uniform_(self.output_weight)

    def forward(
        self,
        features,
        plane_indices,
        slice_mask=None,
    ):
        if (
            features.ndim != 3
            or features.shape[-1] != self.feature_dim
        ):
            raise ValueError(
                "features must have shape [batch, slices, feature_dim]"
            )
        if plane_indices.shape != features.shape[:2]:
            raise ValueError(
                "plane_indices must align one-to-one with features"
            )
        if plane_indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "plane_indices must be an integer tensor"
            )
        if slice_mask is None:
            slice_mask = torch.ones(
                features.shape[:2],
                dtype=torch.bool,
                device=features.device,
            )
        if (
            slice_mask.shape != features.shape[:2]
            or slice_mask.dtype != torch.bool
        ):
            raise ValueError(
                "slice_mask must be boolean with shape [batch, slices]"
            )
        if not torch.all(slice_mask.any(dim=1)):
            raise ValueError(
                "every study must contain at least one valid slice"
            )

        active_planes = plane_indices[slice_mask]
        if (
            torch.any(active_planes < 0)
            or torch.any(active_planes >= self.num_planes)
        ):
            raise ValueError(
                "active plane indices contain an invalid active plane"
            )
        safe_planes = torch.where(
            slice_mask,
            plane_indices,
            torch.zeros_like(plane_indices),
        )

        image_tokens = self.feature_projection(
            features.float()
        )
        plane_tokens = self.plane_embedding(
            safe_planes.long()
        )
        tokens = self.token_fusion(
            torch.cat(
                [image_tokens, plane_tokens],
                dim=-1,
            )
        )

        queries = F.layer_norm(
            self.target_queries,
            normalized_shape=(
                self.target_queries.shape[-1],
            ),
        )
        attention_logits = torch.einsum(
            "bsd,td->bts",
            tokens,
            queries,
        ) * self.attention_scale
        attention_logits = attention_logits.masked_fill(
            ~slice_mask[:, None, :],
            torch.finfo(attention_logits.dtype).min,
        )
        attention_weights = torch.softmax(
            attention_logits,
            dim=-1,
        )
        pooled_targets = torch.einsum(
            "bts,bsd->btd",
            attention_weights,
            tokens,
        )
        target_hidden = self.head_trunk(
            pooled_targets
        )
        logits = torch.einsum(
            "bth,th->bt",
            target_hidden,
            self.output_weight,
        ) + self.output_bias
        expected_shape = (
            features.shape[0],
            self.num_targets,
        )
        if logits.shape != expected_shape:
            raise AssertionError(
                "internal error: unexpected MIL output shape"
            )
        return logits


def weighted_masked_bce(
    logits,
    targets,
    weights,
    pos_weight=None,
):
    if (
        logits.shape != targets.shape
        or logits.shape != weights.shape
    ):
        raise ValueError(
            "logits, targets, and weights must have identical shapes"
        )
    if torch.any(
        torch.isfinite(weights) & (weights < 0)
    ):
        raise ValueError(
            "finite supervision weights cannot be negative"
        )
    if pos_weight is not None:
        if (
            pos_weight.ndim != 1
            or pos_weight.shape[0] != logits.shape[-1]
        ):
            raise ValueError(
                "pos_weight must have one value per target"
            )
        if (
            not torch.all(torch.isfinite(pos_weight))
            or torch.any(pos_weight <= 0)
        ):
            raise ValueError(
                "pos_weight values must be finite and positive"
            )
        pos_weight = pos_weight.to(
            device=logits.device,
            dtype=logits.dtype,
        )

    known = (
        torch.isfinite(targets)
        & torch.isfinite(weights)
        & (weights > 0)
    )
    safe_targets = torch.where(
        known,
        targets,
        torch.zeros_like(targets),
    )
    if torch.any(
        (safe_targets[known] < 0)
        | (safe_targets[known] > 1)
    ):
        raise ValueError(
            "known targets must be probabilities in [0, 1]"
        )
    safe_weights = torch.where(
        known,
        weights,
        torch.zeros_like(weights),
    )
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        safe_targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    numerator = (
        elementwise * safe_weights
    ).sum()
    denominator = safe_weights.sum()
    return numerator / denominator.clamp_min(
        torch.finfo(denominator.dtype).eps
    )


def fold_positive_weight(
    targets,
    weights,
    minimum=0.50,
    maximum=4.00,
):
    target_array = np.asarray(
        targets,
        dtype=np.float32,
    )
    weight_array = np.asarray(
        weights,
        dtype=np.float32,
    )
    if target_array.shape != weight_array.shape:
        raise ValueError(
            "target and weight arrays must have identical shapes"
        )
    known = (
        np.isfinite(target_array)
        & np.isfinite(weight_array)
        & (weight_array > 0)
    )
    safe_targets = np.where(
        known,
        target_array,
        0.0,
    )
    safe_weights = np.where(
        known,
        weight_array,
        0.0,
    )
    positive_mass = (
        safe_weights * safe_targets
    ).sum(axis=0)
    negative_mass = (
        safe_weights * (1.0 - safe_targets)
    ).sum(axis=0)
    both_classes = (
        (positive_mass > 1e-6)
        & (negative_mass > 1e-6)
    )
    ratios = np.ones_like(
        positive_mass,
        dtype=np.float64,
    )
    ratios[both_classes] = (
        negative_mass[both_classes]
        / positive_mass[both_classes]
    )
    ratios = np.clip(
        ratios,
        minimum,
        maximum,
    ).astype(np.float32)
    if (
        not np.isfinite(ratios).all()
        or np.any(ratios <= 0)
    ):
        raise ValueError(
            "invalid fold positive weights"
        )
    return torch.from_numpy(ratios)


class CachedFeatureDatasetV5(Dataset):
    def __init__(
        self,
        features,
        planes,
        masks,
        targets,
        weights,
        row_indices,
    ):
        self.features = features
        self.planes = planes
        self.masks = masks
        self.targets = targets
        self.weights = weights
        raw_rows = np.asarray(row_indices)
        if (
            raw_rows.ndim != 1
            or not np.issubdtype(
                raw_rows.dtype,
                np.integer,
            )
        ):
            raise ValueError(
                "row_indices must be a one-dimensional integer array"
            )
        self.row_indices = raw_rows.astype(
            np.int64,
            copy=True,
        )
        if not (
            features.shape[:2] == planes.shape == masks.shape
            and len(features) == len(targets) == len(weights)
            and targets.shape == weights.shape
        ):
            raise ValueError(
                "cached features and supervision are misaligned"
            )
        if (
            np.any(self.row_indices < 0)
            or np.any(self.row_indices >= len(features))
            or len(np.unique(self.row_indices))
            != len(self.row_indices)
        ):
            raise ValueError(
                "row_indices contain invalid or duplicate rows"
            )

    def __len__(self):
        return len(self.row_indices)

    def __getitem__(self, index):
        row = int(self.row_indices[index])
        mask = self.masks[row].astype(
            bool,
            copy=True,
        )
        plane = self.planes[row].astype(
            np.int64,
            copy=True,
        )
        plane[~mask] = 0
        return {
            "features": torch.from_numpy(
                self.features[row].astype(
                    np.float32,
                    copy=True,
                )
            ),
            "planes": torch.from_numpy(plane),
            "mask": torch.from_numpy(mask),
            "targets": torch.from_numpy(
                self.targets[row].astype(
                    np.float32,
                    copy=True,
                )
            ),
            "weights": torch.from_numpy(
                self.weights[row].astype(
                    np.float32,
                    copy=True,
                )
            ),
        }


def make_cached_loader(
    payload,
    targets,
    weights,
    row_indices,
    *,
    shuffle,
    batch_size,
    seed,
):
    dataset = CachedFeatureDatasetV5(
        payload.features,
        payload.planes,
        payload.masks,
        targets,
        weights,
        row_indices,
    )
    generator = torch.Generator().manual_seed(
        int(seed)
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=bool(torch.cuda.is_available()),
        generator=generator,
    )


    

# %%

# ============================================================
# Version 5 Task 8 RED — cached MIL model, dataset, and loss
# ============================================================
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(20260808)
_test_features = torch.randn(4, 7, 16)
_test_planes = torch.tensor(
    [[0, 0, 1, 1, 2, 2, -1]] * 4,
    dtype=torch.long,
)
_test_mask = torch.tensor(
    [[True, True, True, True, True, True, False]] * 4,
    dtype=torch.bool,
)
_test_model = TargetAttentionMILV3(
    feature_dim=16,
    num_targets=3,
    token_dim=16,
    plane_dim=4,
    hidden_dim=8,
    dropout=0.0,
)
_test_logits = _test_model(_test_features, _test_planes, _test_mask)
assert _test_logits.shape == (4, 3)
assert torch.isfinite(_test_logits).all()

_mutated_features = _test_features.clone()
_mutated_planes = _test_planes.clone()
_mutated_features[~_test_mask] = 12345.0
_mutated_planes[~_test_mask] = -99
_mutated_logits = _test_model(
    _mutated_features,
    _mutated_planes,
    _test_mask,
)
assert torch.allclose(_test_logits, _mutated_logits, atol=1e-6, rtol=0.0)

try:
    _bad_planes = _test_planes.clone()
    _bad_planes[0, 0] = 3
    _test_model(_test_features, _bad_planes, _test_mask)
except ValueError as exc:
    assert "active plane" in str(exc)
else:
    raise AssertionError("invalid active plane was accepted")

try:
    _empty_mask = _test_mask.clone()
    _empty_mask[0] = False
    _test_model(_test_features, _test_planes, _empty_mask)
except ValueError as exc:
    assert "at least one valid" in str(exc)
else:
    raise AssertionError("empty study bag was accepted")

_test_targets = torch.randint(0, 2, _test_logits.shape, dtype=torch.float32)
_test_weights = torch.ones_like(_test_targets)
_test_loss = weighted_masked_bce(
    _test_logits,
    _test_targets,
    _test_weights,
)
_test_loss.backward()
for _name, _parameter in _test_model.named_parameters():
    assert _parameter.grad is not None, _name
    assert torch.isfinite(_parameter.grad).all(), _name

_weight_logits = torch.tensor([[0.2, -0.7, 1.1, -1.3]])
_weight_targets = torch.tensor([[1.0, 0.8, 0.05, 0.0]])
_confidence_weights = torch.tensor([[1.0, 0.70, 0.40, 0.50]])
_weighted_loss = weighted_masked_bce(
    _weight_logits,
    _weight_targets,
    _confidence_weights,
)
_closed_form_bce = (
    torch.clamp(_weight_logits, min=0)
    - _weight_logits * _weight_targets
    + torch.log1p(torch.exp(-torch.abs(_weight_logits)))
)
_expected_weighted_loss = (
    _closed_form_bce * _confidence_weights
).sum() / _confidence_weights.sum()
assert torch.allclose(
    _weighted_loss,
    _expected_weighted_loss,
    atol=1e-7,
    rtol=0.0,
)

_unknown_logits = torch.randn(2, 3, requires_grad=True)
_unknown_loss = weighted_masked_bce(
    _unknown_logits,
    torch.full((2, 3), float("nan")),
    torch.zeros(2, 3),
)
assert _unknown_loss.requires_grad
assert float(_unknown_loss.detach()) == 0.0
_unknown_loss.backward()
assert torch.equal(_unknown_logits.grad, torch.zeros_like(_unknown_logits))

_pos_weight = fold_positive_weight(
    np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    np.ones((2, 2), dtype=np.float32),
)
assert torch.allclose(_pos_weight, torch.tensor([1.0, 1.0]))

_dataset_features = np.zeros((3, 4, 16), dtype=np.float16)
_dataset_planes = np.array([
    [0, 1, 2, -1],
    [2, 2, -1, -1],
    [0, 0, 1, 2],
], dtype=np.int8)
_dataset_masks = _dataset_planes >= 0
_dataset_targets = np.zeros((3, 3), dtype=np.float32)
_dataset_weights = np.ones((3, 3), dtype=np.float32)
_dataset = CachedFeatureDatasetV5(
    _dataset_features,
    _dataset_planes,
    _dataset_masks,
    _dataset_targets,
    _dataset_weights,
    np.array([2, 0], dtype=np.int64),
)
_item = _dataset[1]
assert set(_item) == {
    "features",
    "planes",
    "mask",
    "targets",
    "weights",
}
assert _item["features"].dtype == torch.float32
assert _item["planes"].tolist() == [0, 1, 2, 0]
assert _item["mask"].tolist() == [True, True, True, False]

torch.manual_seed(7)
_fit_features = torch.randn(10, 6, 16)
_fit_planes = torch.tensor(
    [[0, 0, 1, 1, 2, 2]] * 10,
    dtype=torch.long,
)
_fit_mask = torch.ones((10, 6), dtype=torch.bool)
_teacher = torch.randn(16, 3)
_fit_targets = ((_fit_features.mean(dim=1) @ _teacher) > 0).float()
_fit_weights = torch.ones_like(_fit_targets)
_fit_model = TargetAttentionMILV3(
    feature_dim=16,
    num_targets=3,
    token_dim=16,
    plane_dim=4,
    hidden_dim=12,
    dropout=0.0,
)
_fit_optimizer = torch.optim.AdamW(
    _fit_model.parameters(),
    lr=1e-2,
    weight_decay=0.0,
)
with torch.no_grad():
    _initial_fit_loss = float(
        weighted_masked_bce(
            _fit_model(_fit_features, _fit_planes, _fit_mask),
            _fit_targets,
            _fit_weights,
        )
    )
for _fit_step in range(30):
    _fit_optimizer.zero_grad(set_to_none=True)
    _fit_loss = weighted_masked_bce(
        _fit_model(_fit_features, _fit_planes, _fit_mask),
        _fit_targets,
        _fit_weights,
    )
    _fit_loss.backward()
    _fit_optimizer.step()
with torch.no_grad():
    _final_fit_loss = float(
        weighted_masked_bce(
            _fit_model(_fit_features, _fit_planes, _fit_mask),
            _fit_targets,
            _fit_weights,
        )
    )
assert np.isfinite(_final_fit_loss)
assert _final_fit_loss < _initial_fit_loss * 0.50

_parameter_count = sum(
    parameter.numel()
    for parameter in TargetAttentionMILV3().parameters()
)
assert _parameter_count == 122284
print(
    "V5 CACHED MIL CONTRACT PASSED: "
    f"logits={tuple(_test_logits.shape)}, params={_parameter_count}, "
    f"tiny_overfit={_initial_fit_loss:.4f}->{_final_fit_loss:.4f}"
)


    

# %%

# ============================================================
# Version 5 Task 9 GREEN — deterministic fold trainer and exact OOF
# ============================================================
import hashlib
import os
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from sklearn.metrics import roc_auc_score


def _readonly_array_v5(values, *, dtype=None):
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _validated_indices_v5(values, *, n_rows, name):
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must use an integer dtype")
    indices = raw.astype(np.int64, copy=True)
    if len(indices) == 0:
        raise ValueError(f"{name} cannot be empty")
    if np.any(indices < 0) or np.any(indices >= int(n_rows)):
        raise IndexError(f"{name} contains an out-of-range row")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} contains duplicate rows")
    indices.setflags(write=False)
    return indices


def _validated_supervision_matrix_v5(
    values,
    *,
    n_rows,
    n_targets=None,
    name,
):
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != int(n_rows):
        raise ValueError(
            f"{name} must have shape (n_rows, n_targets)"
        )
    if n_targets is not None and matrix.shape[1] != int(n_targets):
        raise ValueError(f"{name} target count mismatch")
    return matrix


def per_target_auc(targets, predictions, weights=None):
    target_array = np.asarray(targets, dtype=np.float64)
    prediction_array = np.asarray(predictions, dtype=np.float64)
    if (
        target_array.ndim != 2
        or prediction_array.shape != target_array.shape
    ):
        raise ValueError(
            "targets and predictions must be equal two-dimensional arrays"
        )
    if not np.all(np.isfinite(prediction_array)):
        raise ValueError("predictions must be finite")
    if np.any(
        (prediction_array < 0.0)
        | (prediction_array > 1.0)
    ):
        raise ValueError("predictions must be probabilities in [0, 1]")

    if weights is None:
        weight_array = np.ones_like(target_array)
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.shape != target_array.shape:
            raise ValueError("AUC weights must match targets")
        if np.any(
            np.isfinite(weight_array)
            & (weight_array < 0.0)
        ):
            raise ValueError("AUC weights cannot be negative")

    scores = np.full(
        target_array.shape[1],
        np.nan,
        dtype=np.float64,
    )
    for target_index in range(target_array.shape[1]):
        known = (
            np.isfinite(target_array[:, target_index])
            & np.isfinite(weight_array[:, target_index])
            & (weight_array[:, target_index] > 0.0)
        )
        labels = target_array[known, target_index]
        if len(labels) == 0:
            continue
        if np.any((labels != 0.0) & (labels != 1.0)):
            raise ValueError("AUC targets must be exactly binary")
        if len(np.unique(labels)) == 2:
            scores[target_index] = roc_auc_score(
                labels,
                prediction_array[known, target_index],
            )
    scores.setflags(write=False)
    return scores


def auc_summary_v5(targets, predictions, weights=None):
    scores = per_target_auc(
        targets,
        predictions,
        weights=weights,
    )
    finite = np.isfinite(scores)
    scored_targets = int(finite.sum())
    macro_auc = (
        float(np.mean(scores[finite]))
        if scored_targets
        else float("nan")
    )
    return MappingProxyType(
        {
            "per_target": scores,
            "macro_auc": macro_auc,
            "scored_targets": scored_targets,
        }
    )


def _average_ranks_v5(values):
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError("rank input must be a finite vector")
    order = np.argsort(vector, kind="mergesort")
    sorted_values = vector[order]
    ranks = np.empty(len(vector), dtype=np.float64)
    start = 0
    while start < len(vector):
        stop = start + 1
        while (
            stop < len(vector)
            and sorted_values[stop] == sorted_values[start]
        ):
            stop += 1
        average_rank = 0.5 * (start + stop - 1)
        ranks[order[start:stop]] = average_rank
        start = stop
    if len(vector) > 1:
        ranks /= float(len(vector) - 1)
    else:
        ranks.fill(0.5)
    return ranks


def rank_average_oof(prediction_matrices):
    """
    Average ranks across complete leakage-free OOF matrices only.

    Each input must already be stitched from held-out predictions for
    every row. Fold models' predictions on their training rows are never
    valid inputs here.
    """
    matrices = [
        np.asarray(matrix, dtype=np.float64)
        for matrix in prediction_matrices
    ]
    if not matrices:
        raise ValueError(
            "at least one complete leakage-free OOF matrix is required"
        )
    shape = matrices[0].shape
    if len(shape) != 2:
        raise ValueError("OOF matrices must be two-dimensional")
    for matrix in matrices:
        if matrix.shape != shape:
            raise ValueError("OOF matrices must share one exact shape")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("OOF matrices must be finite")
        if np.any((matrix < 0.0) | (matrix > 1.0)):
            raise ValueError(
                "OOF matrices must contain probabilities in [0, 1]"
            )

    ranked = np.empty(
        (len(matrices), shape[0], shape[1]),
        dtype=np.float64,
    )
    for matrix_index, matrix in enumerate(matrices):
        for target_index in range(shape[1]):
            ranked[matrix_index, :, target_index] = (
                _average_ranks_v5(matrix[:, target_index])
            )
    averaged = ranked.mean(axis=0)
    averaged.setflags(write=False)
    return averaged


class OOFAccumulator:
    def __init__(self, *, n_rows, n_targets):
        if int(n_rows) <= 0 or int(n_targets) <= 0:
            raise ValueError(
                "OOF dimensions must be positive"
            )
        self._predictions = np.full(
            (int(n_rows), int(n_targets)),
            np.nan,
            dtype=np.float64,
        )
        self._assignment_counts = np.zeros(
            int(n_rows),
            dtype=np.int64,
        )

    @property
    def assignment_counts(self):
        return _readonly_array_v5(
            self._assignment_counts,
            dtype=np.int64,
        )

    def assign(self, row_indices, predictions):
        indices = _validated_indices_v5(
            row_indices,
            n_rows=self._predictions.shape[0],
            name="OOF row_indices",
        )
        values = np.asarray(
            predictions,
            dtype=np.float64,
        )
        expected_shape = (
            len(indices),
            self._predictions.shape[1],
        )
        if values.shape != expected_shape:
            raise ValueError(
                "OOF prediction shape does not match row_indices"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("OOF predictions must be finite")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(
                "OOF predictions must be probabilities in [0, 1]"
            )
        if np.any(self._assignment_counts[indices] != 0):
            raise ValueError(
                "each OOF row may be assigned exactly once"
            )
        self._predictions[indices] = values
        self._assignment_counts[indices] = 1

    def finalize(self):
        if not np.all(self._assignment_counts == 1):
            missing = int(
                np.sum(self._assignment_counts == 0)
            )
            raise ValueError(
                f"OOF accumulator is incomplete: {missing} rows missing"
            )
        if not np.all(np.isfinite(self._predictions)):
            raise ValueError(
                "OOF accumulator contains non-finite predictions"
            )
        return _readonly_array_v5(
            self._predictions,
            dtype=np.float64,
        )


def _validated_sha256_v5(value, *, name):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"{name} must contain only hexadecimal characters"
        ) from error
    return value.lower()


@dataclass(frozen=True)
class CheckpointIdentityV5:
    experiment_id: str
    run_nonce: str
    cache_sha256: str
    fold_assignment_sha256: str
    config_sha256: str
    supervision_sha256: str
    fold_index: int
    seed: int
    target_columns: tuple
    parameter_count: int

    def __post_init__(self):
        for field_name in ("experiment_id", "run_nonce"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string"
                )
        for field_name in (
            "cache_sha256",
            "fold_assignment_sha256",
            "config_sha256",
            "supervision_sha256",
        ):
            _validated_sha256_v5(
                getattr(self, field_name),
                name=field_name,
            )
        if int(self.fold_index) < 0:
            raise ValueError("fold_index cannot be negative")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if (
            not isinstance(self.target_columns, tuple)
            or len(self.target_columns) == 0
            or len(set(self.target_columns))
            != len(self.target_columns)
            or not all(
                isinstance(value, str) and value.strip()
                for value in self.target_columns
            )
        ):
            raise ValueError(
                "target_columns must be a unique non-empty tuple"
            )
        if int(self.parameter_count) <= 0:
            raise ValueError(
                "parameter_count must be positive"
            )


def _checkpoint_identity_dict_v5(identity):
    if not isinstance(identity, CheckpointIdentityV5):
        raise TypeError(
            "checkpoint identity must be CheckpointIdentityV5"
        )
    return {
        "experiment_id": identity.experiment_id,
        "run_nonce": identity.run_nonce,
        "cache_sha256": _validated_sha256_v5(
            identity.cache_sha256,
            name="cache_sha256",
        ),
        "fold_assignment_sha256": _validated_sha256_v5(
            identity.fold_assignment_sha256,
            name="fold_assignment_sha256",
        ),
        "config_sha256": _validated_sha256_v5(
            identity.config_sha256,
            name="config_sha256",
        ),
        "supervision_sha256": _validated_sha256_v5(
            identity.supervision_sha256,
            name="supervision_sha256",
        ),
        "fold_index": int(identity.fold_index),
        "seed": int(identity.seed),
        "target_columns": list(identity.target_columns),
        "parameter_count": int(identity.parameter_count),
    }


def _file_sha256_v5(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _clone_state_dict_cpu_v5(state_dict):
    cloned = {}
    for name, tensor in state_dict.items():
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise TypeError(
                "state_dict must map strings to tensors"
            )
        cpu_tensor = tensor.detach().cpu().clone()
        if not torch.all(torch.isfinite(cpu_tensor)):
            raise ValueError(
                "checkpoint state contains a non-finite tensor"
            )
        cloned[name] = cpu_tensor
    if not cloned:
        raise ValueError("state_dict cannot be empty")
    return cloned


def _atomic_save_checkpoint_v5(
    path,
    *,
    identity,
    state_dict,
    metrics,
):
    target_path = Path(path)
    if target_path.exists():
        raise FileExistsError(
            f"refusing to overwrite checkpoint: {target_path.name}"
        )
    target_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = target_path.with_name(
        f".{target_path.name}.{uuid.uuid4().hex}.tmp"
    )
    payload = {
        "format_version": 1,
        "identity": _checkpoint_identity_dict_v5(identity),
        "metrics": dict(metrics),
        "state_dict": _clone_state_dict_cpu_v5(state_dict),
    }
    try:
        with temporary_path.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return _file_sha256_v5(target_path)


def _load_verified_checkpoint_v5(
    path,
    *,
    expected_identity,
):
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path.name}"
        )
    checkpoint_sha256 = _file_sha256_v5(checkpoint_path)
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict) or set(payload) != {
        "format_version",
        "identity",
        "metrics",
        "state_dict",
    }:
        raise ValueError("checkpoint payload schema mismatch")
    if payload["format_version"] != 1:
        raise ValueError(
            "unsupported checkpoint format version"
        )
    expected = _checkpoint_identity_dict_v5(
        expected_identity
    )
    if payload["identity"] != expected:
        raise ValueError(
            "checkpoint identity does not match the requested run"
        )
    state_dict = _clone_state_dict_cpu_v5(
        payload["state_dict"]
    )
    metrics = payload["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError(
            "checkpoint metrics must be a dictionary"
        )
    return (
        MappingProxyType(state_dict),
        MappingProxyType(dict(metrics)),
        checkpoint_sha256,
    )


@dataclass(frozen=True)
class FoldRunResult:
    predictions: np.ndarray
    validation_indices: np.ndarray
    state_dict: object
    best_epoch: int
    best_score: float
    macro_auc: float
    scored_targets: int
    validation_loss: float
    epochs_ran: int
    train_rows: int
    validation_rows: int
    runtime_seconds: float
    checkpoint_path: object
    checkpoint_sha256: object


def _evaluate_cached_model_v5(
    model,
    loader,
    *,
    device,
    pos_weight,
):
    model.eval()
    prediction_batches = []
    weighted_loss_sum = 0.0
    total_known_weight = 0.0
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(
                device,
                non_blocking=True,
            )
            planes = batch["planes"].to(
                device,
                non_blocking=True,
            )
            mask = batch["mask"].to(
                device,
                non_blocking=True,
            )
            targets = batch["targets"].to(
                device,
                non_blocking=True,
            )
            weights = batch["weights"].to(
                device,
                non_blocking=True,
            )
            logits = model(features, planes, mask)
            loss = weighted_masked_bce(
                logits,
                targets,
                weights,
                pos_weight=pos_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "validation loss became non-finite"
                )
            known_weight = torch.where(
                torch.isfinite(targets)
                & torch.isfinite(weights)
                & (weights > 0),
                weights,
                torch.zeros_like(weights),
            ).sum()
            batch_known_weight = float(
                known_weight.detach().cpu()
            )
            weighted_loss_sum += (
                float(loss.detach().cpu())
                * batch_known_weight
            )
            total_known_weight += batch_known_weight
            probabilities = torch.sigmoid(logits)
            if not torch.all(torch.isfinite(probabilities)):
                raise FloatingPointError(
                    "validation predictions became non-finite"
                )
            prediction_batches.append(
                probabilities.detach().cpu().numpy()
            )
    if not prediction_batches or total_known_weight <= 0.0:
        raise ValueError(
            "validation loader has no scored supervision"
        )
    predictions = np.concatenate(
        prediction_batches,
        axis=0,
    ).astype(np.float64, copy=False)
    validation_loss = (
        weighted_loss_sum / total_known_weight
    )
    return predictions, float(validation_loss)


def train_fold_model_v5(
    *,
    payload,
    training_targets,
    training_weights,
    validation_targets,
    validation_weights,
    train_indices,
    validation_indices,
    model_factory,
    device,
    epochs,
    patience,
    batch_size,
    seed,
    learning_rate=3e-4,
    weight_decay=1e-3,
    grad_clip=1.0,
    checkpoint_path=None,
    checkpoint_identity=None,
):
    started_at = time.perf_counter()
    if not callable(model_factory):
        raise TypeError("model_factory must be callable")
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    if int(patience) <= 0:
        raise ValueError("patience must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if float(weight_decay) < 0.0:
        raise ValueError("weight_decay cannot be negative")
    if float(grad_clip) <= 0.0:
        raise ValueError("grad_clip must be positive")
    if (checkpoint_path is None) != (
        checkpoint_identity is None
    ):
        raise ValueError(
            "checkpoint_path and checkpoint_identity must be supplied together"
        )

    feature_array = np.asarray(payload.features)
    plane_array = np.asarray(payload.planes)
    mask_array = np.asarray(payload.masks)
    if feature_array.ndim != 3:
        raise ValueError(
            "cached features must be three-dimensional"
        )
    n_rows = feature_array.shape[0]
    if (
        plane_array.shape != feature_array.shape[:2]
        or mask_array.shape != feature_array.shape[:2]
    ):
        raise ValueError(
            "cached planes and masks must match cached features"
        )

    train_candidates = _validated_indices_v5(
        train_indices,
        n_rows=n_rows,
        name="train_indices",
    )
    valid_rows = _validated_indices_v5(
        validation_indices,
        n_rows=n_rows,
        name="validation_indices",
    )
    if np.intersect1d(
        train_candidates,
        valid_rows,
    ).size:
        raise ValueError(
            "training and validation rows must be disjoint"
        )

    train_target_array = _validated_supervision_matrix_v5(
        training_targets,
        n_rows=n_rows,
        name="training_targets",
    )
    n_targets = train_target_array.shape[1]
    train_weight_array = _validated_supervision_matrix_v5(
        training_weights,
        n_rows=n_rows,
        n_targets=n_targets,
        name="training_weights",
    )
    valid_target_array = _validated_supervision_matrix_v5(
        validation_targets,
        n_rows=n_rows,
        n_targets=n_targets,
        name="validation_targets",
    )
    valid_weight_array = _validated_supervision_matrix_v5(
        validation_weights,
        n_rows=n_rows,
        n_targets=n_targets,
        name="validation_weights",
    )
    for matrix_name, matrix in (
        ("training_weights", train_weight_array),
        ("validation_weights", valid_weight_array),
    ):
        if np.any(
            np.isfinite(matrix) & (matrix < 0.0)
        ):
            raise ValueError(
                f"{matrix_name} cannot contain negative values"
            )

    active_rows = mask_array.any(axis=1)
    if not np.all(active_rows[valid_rows]):
        raise ValueError(
            "every validation row must contain cached features"
        )
    heldout_training_weight = (
        np.isfinite(train_weight_array[valid_rows])
        & (train_weight_array[valid_rows] > 0.0)
    )
    if np.any(heldout_training_weight):
        raise ValueError(
            "held-out rows must have zero training weight"
        )

    train_known = (
        np.isfinite(train_target_array)
        & np.isfinite(train_weight_array)
        & (train_weight_array > 0.0)
    )
    keep_train = (
        active_rows[train_candidates]
        & train_known[train_candidates].any(axis=1)
    )
    effective_train_rows = train_candidates[keep_train]
    if len(effective_train_rows) == 0:
        raise ValueError(
            "no usable training rows remain"
        )
    if np.intersect1d(
        effective_train_rows,
        valid_rows,
    ).size:
        raise AssertionError(
            "held-out leakage reached the training loader"
        )
    valid_known = (
        np.isfinite(valid_target_array[valid_rows])
        & np.isfinite(valid_weight_array[valid_rows])
        & (valid_weight_array[valid_rows] > 0.0)
    )
    if not valid_known.any():
        raise ValueError(
            "validation rows contain no official supervision"
        )

    compute_device = torch.device(device)
    if (
        compute_device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable"
        )
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = model_factory().to(compute_device)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    if parameter_count <= 0:
        raise ValueError("model has no trainable parameters")
    if checkpoint_identity is not None:
        if checkpoint_identity.seed != seed:
            raise ValueError(
                "checkpoint seed does not match trainer seed"
            )
        if (
            checkpoint_identity.parameter_count
            != parameter_count
        ):
            raise ValueError(
                "checkpoint parameter count does not match model"
            )
        if (
            len(checkpoint_identity.target_columns)
            != n_targets
        ):
            raise ValueError(
                "checkpoint target schema does not match supervision"
            )

    train_loader = make_cached_loader(
        payload,
        train_target_array,
        train_weight_array,
        effective_train_rows,
        shuffle=True,
        batch_size=int(batch_size),
        seed=seed,
    )
    valid_loader = make_cached_loader(
        payload,
        valid_target_array,
        valid_weight_array,
        valid_rows,
        shuffle=False,
        batch_size=int(batch_size),
        seed=seed,
    )
    np.testing.assert_array_equal(
        np.asarray(train_loader.dataset.row_indices),
        effective_train_rows,
    )
    np.testing.assert_array_equal(
        np.asarray(valid_loader.dataset.row_indices),
        valid_rows,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    pos_weight = fold_positive_weight(
        train_target_array[effective_train_rows],
        train_weight_array[effective_train_rows],
    ).to(
        compute_device,
        dtype=torch.float32,
    )

    best_score = float("-inf")
    best_state = None
    best_epoch = 0
    best_macro_auc = float("nan")
    best_scored_targets = 0
    best_validation_loss = float("inf")
    bad_epochs = 0
    epochs_ran = 0

    for epoch in range(1, int(epochs) + 1):
        epochs_ran = epoch
        model.train()
        for batch in train_loader:
            features = batch["features"].to(
                compute_device,
                non_blocking=True,
            )
            planes = batch["planes"].to(
                compute_device,
                non_blocking=True,
            )
            mask = batch["mask"].to(
                compute_device,
                non_blocking=True,
            )
            targets = batch["targets"].to(
                compute_device,
                non_blocking=True,
            )
            weights = batch["weights"].to(
                compute_device,
                non_blocking=True,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, planes, mask)
            loss = weighted_masked_bce(
                logits,
                targets,
                weights,
                pos_weight=pos_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "training loss became non-finite"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(grad_clip),
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    "gradient norm became non-finite"
                )
            optimizer.step()

        validation_predictions, validation_loss = (
            _evaluate_cached_model_v5(
                model,
                valid_loader,
                device=compute_device,
                pos_weight=pos_weight,
            )
        )
        if validation_predictions.shape != (
            len(valid_rows),
            n_targets,
        ):
            raise AssertionError(
                "validation prediction shape drifted"
            )
        summary = auc_summary_v5(
            valid_target_array[valid_rows],
            validation_predictions,
            weights=valid_weight_array[valid_rows],
        )
        macro_auc = float(summary["macro_auc"])
        scored_targets = int(
            summary["scored_targets"]
        )
        score = (
            macro_auc
            if (
                scored_targets >= 3
                and np.isfinite(macro_auc)
            )
            else -float(validation_loss)
        )

        if score > best_score:
            best_score = float(score)
            best_state = _clone_state_dict_cpu_v5(
                model.state_dict()
            )
            best_epoch = int(epoch)
            best_macro_auc = macro_auc
            best_scored_targets = scored_targets
            best_validation_loss = float(
                validation_loss
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break

    if best_state is None or best_epoch == 0:
        raise RuntimeError(
            "training did not produce a valid checkpoint"
        )

    checkpoint_sha256 = None
    resolved_checkpoint_path = None
    if checkpoint_path is not None:
        checkpoint_metrics = {
            "best_epoch": best_epoch,
            "best_score": best_score,
            "macro_auc": best_macro_auc,
            "scored_targets": best_scored_targets,
            "validation_loss": best_validation_loss,
            "epochs_ran": epochs_ran,
            "train_rows": int(len(effective_train_rows)),
            "validation_rows": int(len(valid_rows)),
        }
        checkpoint_sha256 = _atomic_save_checkpoint_v5(
            checkpoint_path,
            identity=checkpoint_identity,
            state_dict=best_state,
            metrics=checkpoint_metrics,
        )
        (
            loaded_state,
            loaded_metrics,
            loaded_sha256,
        ) = _load_verified_checkpoint_v5(
            checkpoint_path,
            expected_identity=checkpoint_identity,
        )
        if checkpoint_sha256 != loaded_sha256:
            raise AssertionError(
                "checkpoint SHA changed during verified reload"
            )
        if int(loaded_metrics["best_epoch"]) != best_epoch:
            raise AssertionError(
                "checkpoint best epoch changed during reload"
            )
        prediction_state = dict(loaded_state)
        resolved_checkpoint_path = str(
            Path(checkpoint_path)
        )
    else:
        prediction_state = _clone_state_dict_cpu_v5(
            best_state
        )

    prediction_model = model_factory().to(
        compute_device
    )
    prediction_model.load_state_dict(
        prediction_state,
        strict=True,
    )
    (
        final_predictions,
        final_validation_loss,
    ) = _evaluate_cached_model_v5(
        prediction_model,
        valid_loader,
        device=compute_device,
        pos_weight=pos_weight,
    )
    if final_predictions.shape != (
        len(valid_rows),
        n_targets,
    ):
        raise AssertionError(
            "reloaded checkpoint prediction shape drifted"
        )
    final_summary = auc_summary_v5(
        valid_target_array[valid_rows],
        final_predictions,
        weights=valid_weight_array[valid_rows],
    )
    final_macro_auc = float(
        final_summary["macro_auc"]
    )
    final_scored_targets = int(
        final_summary["scored_targets"]
    )
    if final_scored_targets != best_scored_targets:
        raise AssertionError(
            "reloaded checkpoint scorable target count changed"
        )
    if (
        np.isfinite(best_macro_auc)
        != np.isfinite(final_macro_auc)
    ) or (
        np.isfinite(best_macro_auc)
        and not np.isclose(
            best_macro_auc,
            final_macro_auc,
            rtol=0.0,
            atol=1e-10,
        )
    ):
        raise AssertionError(
            "reloaded checkpoint macro AUC changed"
        )
    if not np.isclose(
        best_validation_loss,
        final_validation_loss,
        rtol=0.0,
        atol=1e-8,
    ):
        raise AssertionError(
            "reloaded checkpoint validation loss changed"
        )

    final_predictions = _readonly_array_v5(
        final_predictions,
        dtype=np.float64,
    )
    final_validation_indices = _readonly_array_v5(
        valid_rows,
        dtype=np.int64,
    )
    final_state = MappingProxyType(
        _clone_state_dict_cpu_v5(
            prediction_model.state_dict()
        )
    )
    return FoldRunResult(
        predictions=final_predictions,
        validation_indices=final_validation_indices,
        state_dict=final_state,
        best_epoch=best_epoch,
        best_score=best_score,
        macro_auc=final_macro_auc,
        scored_targets=final_scored_targets,
        validation_loss=float(
            final_validation_loss
        ),
        epochs_ran=epochs_ran,
        train_rows=int(len(effective_train_rows)),
        validation_rows=int(len(valid_rows)),
        runtime_seconds=float(
            time.perf_counter() - started_at
        ),
        checkpoint_path=resolved_checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    )


    

# %%

# ============================================================
# Version 5 Task 9 RED/GREEN — deterministic trainer and OOF contract
# ============================================================
from pathlib import Path
from types import MappingProxyType

_tiny_rng = np.random.default_rng(9127)
_tiny_features = _tiny_rng.normal(
    size=(20, 6, 16)
).astype(np.float16)
_tiny_planes = np.tile(
    np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int8),
    (20, 1),
)
_tiny_masks = np.ones((20, 6), dtype=bool)
for _tiny_array in (
    _tiny_features,
    _tiny_planes,
    _tiny_masks,
):
    _tiny_array.setflags(write=False)

_tiny_payload = FeaturePayload(
    features=_tiny_features,
    planes=_tiny_planes,
    masks=_tiny_masks,
    counters=MappingProxyType(
        {
            "decode_failures": 0,
            "missing_planes": 0,
            "fallback_studies": 0,
            "no_series_studies": 0,
        }
    ),
    elapsed_seconds=1.0,
    cache_sha256="a" * 64,
    source_script_version=340880172,
    source_run_nonce="synthetic-contract",
    slices_per_plane=2,
    row_order_evidence="synthetic-fixed-order",
)

_tiny_signal = _tiny_features.astype(np.float32).mean(axis=1)
_tiny_logits = np.stack(
    (
        _tiny_signal[:, 0] + 0.5 * _tiny_signal[:, 1],
        -_tiny_signal[:, 2] + 0.4 * _tiny_signal[:, 3],
        _tiny_signal[:, 4] - 0.3 * _tiny_signal[:, 5],
    ),
    axis=1,
)
_tiny_labels = (
    _tiny_logits
    > np.median(_tiny_logits, axis=0, keepdims=True)
).astype(np.float32)
_tiny_labels[16:] = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

_tiny_train_indices = np.arange(16, dtype=np.int64)
_tiny_valid_indices = np.arange(16, 20, dtype=np.int64)
_tiny_training_targets = np.full(
    (20, 3),
    np.nan,
    dtype=np.float32,
)
_tiny_training_weights = np.zeros(
    (20, 3),
    dtype=np.float32,
)
_tiny_training_targets[_tiny_train_indices] = _tiny_labels[
    _tiny_train_indices
]
_tiny_training_weights[_tiny_train_indices] = 1.0

_tiny_validation_targets = np.full(
    (20, 3),
    np.nan,
    dtype=np.float32,
)
_tiny_validation_weights = np.zeros(
    (20, 3),
    dtype=np.float32,
)
_tiny_validation_targets[_tiny_valid_indices] = _tiny_labels[
    _tiny_valid_indices
]
_tiny_validation_weights[_tiny_valid_indices] = 1.0

def _tiny_model_factory():
    return TargetAttentionMILV3(
        feature_dim=16,
        num_targets=3,
        token_dim=16,
        plane_dim=4,
        hidden_dim=12,
        dropout=0.0,
    )

_tiny_parameter_count = sum(
    parameter.numel()
    for parameter in _tiny_model_factory().parameters()
)

def _tiny_checkpoint_identity(run_nonce):
    return CheckpointIdentityV5(
        experiment_id="task9_contract",
        run_nonce=run_nonce,
        cache_sha256="a" * 64,
        fold_assignment_sha256="b" * 64,
        config_sha256="c" * 64,
        supervision_sha256="d" * 64,
        fold_index=0,
        seed=123,
        target_columns=("t0", "t1", "t2"),
        parameter_count=_tiny_parameter_count,
    )

_checkpoint_paths = tuple(
    Path(f"/kaggle/working/v5_task9_contract_{index}.pt")
    for index in range(3)
)
for _checkpoint_path in _checkpoint_paths:
    _checkpoint_path.unlink(missing_ok=True)

_common_train_kwargs = dict(
    payload=_tiny_payload,
    training_targets=_tiny_training_targets,
    training_weights=_tiny_training_weights,
    validation_targets=_tiny_validation_targets,
    validation_weights=_tiny_validation_weights,
    train_indices=_tiny_train_indices,
    validation_indices=_tiny_valid_indices,
    model_factory=_tiny_model_factory,
    device="cpu",
    epochs=4,
    patience=2,
    batch_size=4,
    seed=123,
)

_result_a = train_fold_model_v5(
    **_common_train_kwargs,
    checkpoint_path=_checkpoint_paths[0],
    checkpoint_identity=_tiny_checkpoint_identity("repeat-a"),
)
_result_b = train_fold_model_v5(
    **_common_train_kwargs,
    checkpoint_path=_checkpoint_paths[1],
    checkpoint_identity=_tiny_checkpoint_identity("repeat-b"),
)

assert _result_a.predictions.shape == (4, 3)
assert np.all(np.isfinite(_result_a.predictions))
assert np.all(
    (0.0 <= _result_a.predictions)
    & (_result_a.predictions <= 1.0)
)
assert not _result_a.predictions.flags.writeable
assert not _result_a.validation_indices.flags.writeable
np.testing.assert_array_equal(
    _result_a.validation_indices,
    _tiny_valid_indices,
)
np.testing.assert_allclose(
    _result_a.predictions,
    _result_b.predictions,
    rtol=0.0,
    atol=1e-7,
)
assert _result_a.train_rows == 16
assert _result_a.validation_rows == 4
assert 1 <= _result_a.best_epoch <= _result_a.epochs_ran <= 4
assert _result_a.scored_targets == 3
assert np.isfinite(_result_a.macro_auc)
assert len(_result_a.checkpoint_sha256) == 64
assert all(
    tensor.device.type == "cpu"
    and torch.all(torch.isfinite(tensor))
    for tensor in _result_a.state_dict.values()
)

_heldout_mutated_targets = _tiny_training_targets.copy()
_heldout_mutated_targets[_tiny_valid_indices] = 1.0
_result_heldout_mutated = train_fold_model_v5(
    **{
        **_common_train_kwargs,
        "training_targets": _heldout_mutated_targets,
    },
    checkpoint_path=_checkpoint_paths[2],
    checkpoint_identity=_tiny_checkpoint_identity("heldout-mutated"),
)
np.testing.assert_allclose(
    _result_a.predictions,
    _result_heldout_mutated.predictions,
    rtol=0.0,
    atol=1e-7,
)

_expected_oof = np.asarray(
    [
        [0.10, 0.40, 0.90],
        [0.30, 0.20, 0.70],
        [0.80, 0.60, 0.10],
        [0.50, 0.90, 0.30],
    ],
    dtype=np.float64,
)
_oof = OOFAccumulator(n_rows=4, n_targets=3)
_oof.assign(
    np.asarray([2, 0], dtype=np.int64),
    _expected_oof[[2, 0]],
)
_oof.assign(
    np.asarray([1, 3], dtype=np.int64),
    _expected_oof[[1, 3]],
)
_final_oof = _oof.finalize()
np.testing.assert_allclose(_final_oof, _expected_oof)
assert not _final_oof.flags.writeable
np.testing.assert_array_equal(
    _oof.assignment_counts,
    np.ones(4, dtype=np.int64),
)

_duplicate_oof = OOFAccumulator(n_rows=2, n_targets=3)
_duplicate_oof.assign(
    np.asarray([0], dtype=np.int64),
    np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
)
try:
    _duplicate_oof.assign(
        np.asarray([0], dtype=np.int64),
        np.asarray([[0.4, 0.5, 0.6]], dtype=np.float64),
    )
    raise AssertionError("duplicate OOF assignment was accepted")
except ValueError:
    pass
try:
    _duplicate_oof.finalize()
    raise AssertionError("incomplete OOF accumulator was accepted")
except ValueError:
    pass
try:
    OOFAccumulator(n_rows=1, n_targets=3).assign(
        np.asarray([0], dtype=np.int64),
        np.asarray([[0.1, np.nan, 0.3]], dtype=np.float64),
    )
    raise AssertionError("non-finite OOF predictions were accepted")
except ValueError:
    pass

_auc_targets = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_auc_summary = auc_summary_v5(
    _auc_targets,
    _expected_oof,
)
assert _auc_summary["scored_targets"] == 3
assert np.isfinite(_auc_summary["macro_auc"])
assert not _auc_summary["per_target"].flags.writeable

_rank_once = rank_average_oof([_expected_oof])
_rank_twice = rank_average_oof(
    [_expected_oof, np.sqrt(_expected_oof)]
)
np.testing.assert_allclose(_rank_once, _rank_twice)
assert not _rank_twice.flags.writeable

for _checkpoint_path in _checkpoint_paths:
    assert _checkpoint_path.is_file()
    _checkpoint_path.unlink()

print(
    "V5 TRAINER OOF CONTRACT PASSED: "
    f"predictions={_result_a.predictions.shape}, "
    f"best_epoch={_result_a.best_epoch}, "
    f"macro_auc={_result_a.macro_auc:.4f}, "
    "deterministic=True, heldout_immutable=True"
)


    

# %%

# ============================================================
# Version 5 Task 10 GREEN — leakage-safe experiment runner
# ============================================================
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


def _canonical_sha256_v5(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256_v5(values):
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "shape": list(array.shape),
                "dtype": array.dtype.str,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _experiment_config_payload_v5(
    config,
    *,
    batch_size,
    grad_clip,
):
    if not isinstance(config, ExperimentConfig):
        raise TypeError(
            "config must be a registered ExperimentConfig"
        )
    return {
        "name": config.name,
        "model_kind": config.model_kind,
        "supervision_policy": config.supervision_policy,
        "gold_multiplier": float(
            config.gold_multiplier
        ),
        "seeds": [int(seed) for seed in config.seeds],
        "epochs": int(config.epochs),
        "patience": int(config.patience),
        "learning_rate": float(
            config.learning_rate
        ),
        "weight_decay": float(
            config.weight_decay
        ),
        "batch_size": int(batch_size),
        "grad_clip": float(grad_clip),
        "selection": (
            "macro_auc_if_at_least_3_targets_else_negative_loss"
        ),
        "seed_rule": "base_seed_plus_fold_index",
        "precision": "fp32",
        "sampler": "persistent_seeded_shuffle",
    }


def _supervision_sha256_v5(
    bundle,
    *,
    training_gold_rows,
    heldout_gold_rows,
):
    payload = {
        "policy": bundle.policy,
        "gold_multiplier": float(
            bundle.gold_multiplier
        ),
        "enabled_weak_targets": (
            np.asarray(
                bundle.enabled_weak_targets,
                dtype=np.uint8,
            ).tolist()
        ),
        "values_sha256": _array_sha256_v5(
            bundle.values
        ),
        "weights_sha256": _array_sha256_v5(
            bundle.weights
        ),
        "training_gold_rows_sha256": _array_sha256_v5(
            np.asarray(
                training_gold_rows,
                dtype=np.int64,
            )
        ),
        "heldout_gold_rows_sha256": _array_sha256_v5(
            np.asarray(
                heldout_gold_rows,
                dtype=np.int64,
            )
        ),
    }
    return _canonical_sha256_v5(payload)


@dataclass(frozen=True)
class ExperimentResultV5:
    config_name: str
    config_sha256: str
    run_nonce: str
    cache_sha256: str
    source_script_version: int
    fold_assignment_sha256: str
    train_order_sha256: str
    train_csv_sha256: str
    weak_label_sha256: str
    supervision_sha256s: tuple
    fold_enabled_counts: tuple
    seeds: tuple
    oof_predictions: np.ndarray
    per_target_auc: np.ndarray
    macro_auc: float
    scored_targets: int
    assignment_counts: np.ndarray
    fold_records: tuple
    runtime_seconds: float


def _validate_runner_contract_v5(
    *,
    config,
    payload,
    fold_contract,
    report_supervision,
    model_factory,
    run_nonce,
    batch_size,
    grad_clip,
):
    if not isinstance(config, ExperimentConfig):
        raise TypeError(
            "config must be a registered ExperimentConfig"
        )
    if get_experiment(config.name) != config:
        raise ValueError(
            "experiment config is not the immutable registry entry"
        )
    if config.model_kind != "v3_attention":
        raise NotImplementedError(
            "only the tested V3 attention head is enabled"
        )
    if (
        not isinstance(config.seeds, tuple)
        or not config.seeds
        or len(set(config.seeds)) != len(config.seeds)
        or not all(
            isinstance(seed, int)
            for seed in config.seeds
        )
    ):
        raise ValueError(
            "experiment seeds must be a unique integer tuple"
        )
    if not isinstance(run_nonce, str) or not run_nonce.strip():
        raise ValueError(
            "run_nonce must be a non-empty string"
        )
    if int(batch_size) <= 0:
        raise ValueError(
            "batch_size must be positive"
        )
    if float(grad_clip) <= 0.0:
        raise ValueError(
            "grad_clip must be positive"
        )
    if not callable(model_factory):
        raise TypeError(
            "model_factory must be callable"
        )

    features = np.asarray(payload.features)
    masks = np.asarray(payload.masks)
    if (
        features.shape != (
            len(TRAIN_DF),
            CACHE_SHAPE[1],
            CACHE_SHAPE[2],
        )
        or masks.shape != features.shape[:2]
        or not np.all(masks.any(axis=1))
    ):
        raise ValueError(
            "runner requires the complete verified Version 3 cache"
        )
    if (
        payload.cache_sha256
        != TRAIN_CACHE_CONTRACT.expected_sha256
        or payload.source_script_version
        != TRAIN_CACHE_CONTRACT.source_script_version
    ):
        raise ValueError(
            "runner cache provenance mismatch"
        )
    if (
        fold_contract.assignment_sha256
        != FOLD_CONTRACT.assignment_sha256
        or fold_contract.study_order_sha256
        != FOLD_CONTRACT.study_order_sha256
        or fold_contract.train_csv_sha256
        != FOLD_CONTRACT.train_csv_sha256
    ):
        raise ValueError(
            "runner fold provenance mismatch"
        )
    if (
        report_supervision.weak_digest
        != REPORT_SUPERVISION.weak_digest
    ):
        raise ValueError(
            "runner weak-label provenance mismatch"
        )

    official = np.asarray(
        report_supervision.official_values,
        dtype=np.float32,
    )
    if official.shape != (
        len(TRAIN_DF),
        len(TARGETS),
    ):
        raise ValueError(
            "official supervision shape mismatch"
        )
    gold_positions = np.asarray(
        fold_contract.gold_positions,
        dtype=np.int64,
    )
    gold_values = np.asarray(
        fold_contract.gold_values,
        dtype=np.float32,
    )
    if (
        len(gold_positions) != 58
        or len(np.unique(gold_positions)) != 58
        or np.any(gold_positions < 0)
        or np.any(gold_positions >= len(TRAIN_DF))
        or gold_values.shape != (58, len(TARGETS))
    ):
        raise ValueError(
            "gold-row contract mismatch"
        )
    np.testing.assert_array_equal(
        official[gold_positions],
        gold_values,
    )

    validation_seen = np.zeros(
        len(gold_positions),
        dtype=np.int64,
    )
    expected_relative = np.arange(
        len(gold_positions),
        dtype=np.int64,
    )
    if (
        len(fold_contract.folds)
        != fold_contract.n_folds
        or fold_contract.n_folds != 5
    ):
        raise ValueError(
            "runner requires the pinned five-fold split"
        )
    for train_relative, valid_relative in (
        fold_contract.folds
    ):
        train_relative = _validated_indices_v5(
            train_relative,
            n_rows=len(gold_positions),
            name="fold train-relative rows",
        )
        valid_relative = _validated_indices_v5(
            valid_relative,
            n_rows=len(gold_positions),
            name="fold validation-relative rows",
        )
        if np.intersect1d(
            train_relative,
            valid_relative,
        ).size:
            raise ValueError(
                "fold train and validation rows overlap"
            )
        np.testing.assert_array_equal(
            np.sort(
                np.concatenate(
                    (train_relative, valid_relative)
                )
            ),
            expected_relative,
        )
        validation_seen[valid_relative] += 1
    if not np.all(validation_seen == 1):
        raise ValueError(
            "fold validation rows must partition all gold rows exactly once"
        )


def run_oof_experiment_v5(
    *,
    config,
    payload,
    fold_contract,
    report_supervision,
    model_factory,
    device,
    checkpoint_root,
    run_nonce,
    batch_size=32,
    grad_clip=1.0,
    trainer=train_fold_model_v5,
):
    started_at = time.perf_counter()
    _validate_runner_contract_v5(
        config=config,
        payload=payload,
        fold_contract=fold_contract,
        report_supervision=report_supervision,
        model_factory=model_factory,
        run_nonce=run_nonce,
        batch_size=batch_size,
        grad_clip=grad_clip,
    )
    if not callable(trainer):
        raise TypeError("trainer must be callable")

    config_payload = _experiment_config_payload_v5(
        config,
        batch_size=batch_size,
        grad_clip=grad_clip,
    )
    config_sha256 = _canonical_sha256_v5(
        config_payload
    )
    checkpoint_root = Path(checkpoint_root)
    all_global_rows = np.arange(
        len(TRAIN_DF),
        dtype=np.int64,
    )
    parameter_count = sum(
        parameter.numel()
        for parameter in model_factory().parameters()
    )
    if parameter_count != 122284:
        raise ValueError(
            "registered V3 attention model parameter count drifted"
        )

    oof = OOFAccumulator(
        n_rows=len(fold_contract.gold_positions),
        n_targets=len(TARGETS),
    )
    supervision_sha256s = []
    enabled_counts = []
    fold_records = []

    for fold_index, (
        train_relative,
        valid_relative,
    ) in enumerate(fold_contract.folds):
        train_relative = np.asarray(
            train_relative,
            dtype=np.int64,
        )
        valid_relative = np.asarray(
            valid_relative,
            dtype=np.int64,
        )
        training_gold_global = (
            fold_contract.gold_positions[
                train_relative
            ]
        )
        validation_gold_global = (
            fold_contract.gold_positions[
                valid_relative
            ]
        )
        training_global_rows = np.setdiff1d(
            all_global_rows,
            validation_gold_global,
            assume_unique=True,
        )
        bundle = build_fold_supervision(
            report_supervision.weak_values,
            report_supervision.weak_weights,
            report_supervision.official_values,
            report_supervision.structural_enabled,
            training_gold_global,
            validation_gold_global,
            policy=config.supervision_policy,
            gold_multiplier=config.gold_multiplier,
        )
        if np.any(
            np.isfinite(
                bundle.weights[
                    validation_gold_global
                ]
            )
            & (
                bundle.weights[
                    validation_gold_global
                ] > 0
            )
        ):
            raise AssertionError(
                "held-out supervision was not erased"
            )
        np.testing.assert_array_equal(
            bundle.values[training_gold_global],
            report_supervision.official_values[
                training_gold_global
            ],
        )
        np.testing.assert_array_equal(
            bundle.weights[training_gold_global],
            np.full(
                (
                    len(training_gold_global),
                    len(TARGETS),
                ),
                float(config.gold_multiplier),
                dtype=np.float32,
            ),
        )

        validation_targets = np.full(
            (len(TRAIN_DF), len(TARGETS)),
            np.nan,
            dtype=np.float32,
        )
        validation_weights = np.zeros(
            (len(TRAIN_DF), len(TARGETS)),
            dtype=np.float32,
        )
        validation_targets[
            validation_gold_global
        ] = report_supervision.official_values[
            validation_gold_global
        ]
        validation_weights[
            validation_gold_global
        ] = 1.0

        supervision_sha256 = (
            _supervision_sha256_v5(
                bundle,
                training_gold_rows=training_gold_global,
                heldout_gold_rows=validation_gold_global,
            )
        )
        supervision_sha256s.append(
            supervision_sha256
        )
        enabled_counts.append(
            int(
                np.asarray(
                    bundle.enabled_weak_targets,
                    dtype=bool,
                ).sum()
            )
        )

        seed_predictions = []
        for base_seed in config.seeds:
            actual_seed = int(base_seed) + fold_index
            checkpoint_name = (
                f"{config.name}__{run_nonce}"
                f"__fold{fold_index}"
                f"__seed{actual_seed}.pt"
            )
            checkpoint_path = (
                checkpoint_root / checkpoint_name
            )
            identity = CheckpointIdentityV5(
                experiment_id=config.name,
                run_nonce=run_nonce,
                cache_sha256=payload.cache_sha256,
                fold_assignment_sha256=(
                    fold_contract.assignment_sha256
                ),
                config_sha256=config_sha256,
                supervision_sha256=(
                    supervision_sha256
                ),
                fold_index=fold_index,
                seed=actual_seed,
                target_columns=tuple(TARGETS),
                parameter_count=parameter_count,
            )
            fold_result = trainer(
                payload=payload,
                training_targets=bundle.values,
                training_weights=bundle.weights,
                validation_targets=validation_targets,
                validation_weights=validation_weights,
                train_indices=training_global_rows,
                validation_indices=(
                    validation_gold_global
                ),
                model_factory=model_factory,
                device=device,
                epochs=config.epochs,
                patience=config.patience,
                batch_size=batch_size,
                seed=actual_seed,
                learning_rate=(
                    config.learning_rate
                ),
                weight_decay=config.weight_decay,
                grad_clip=grad_clip,
                checkpoint_path=checkpoint_path,
                checkpoint_identity=identity,
            )
            np.testing.assert_array_equal(
                fold_result.validation_indices,
                validation_gold_global,
            )
            if fold_result.predictions.shape != (
                len(validation_gold_global),
                len(TARGETS),
            ):
                raise AssertionError(
                    "fold prediction shape drifted"
                )
            seed_predictions.append(
                np.asarray(
                    fold_result.predictions,
                    dtype=np.float64,
                )
            )
            fold_records.append(
                MappingProxyType(
                    {
                        "fold_index": fold_index,
                        "seed": actual_seed,
                        "best_epoch": int(
                            fold_result.best_epoch
                        ),
                        "best_score": float(
                            fold_result.best_score
                        ),
                        "macro_auc": float(
                            fold_result.macro_auc
                        ),
                        "scored_targets": int(
                            fold_result.scored_targets
                        ),
                        "validation_loss": float(
                            fold_result.validation_loss
                        ),
                        "epochs_ran": int(
                            fold_result.epochs_ran
                        ),
                        "train_rows": int(
                            fold_result.train_rows
                        ),
                        "validation_rows": int(
                            fold_result.validation_rows
                        ),
                        "runtime_seconds": float(
                            fold_result.runtime_seconds
                        ),
                        "checkpoint_sha256": (
                            fold_result.checkpoint_sha256
                        ),
                    }
                )
            )

        averaged_fold_predictions = np.mean(
            np.stack(
                seed_predictions,
                axis=0,
            ),
            axis=0,
        )
        if not np.all(
            np.isfinite(
                averaged_fold_predictions
            )
        ):
            raise FloatingPointError(
                "seed-averaged fold predictions are non-finite"
            )
        oof.assign(
            valid_relative,
            averaged_fold_predictions,
        )

    oof_predictions = oof.finalize()
    score_summary = auc_summary_v5(
        fold_contract.gold_values,
        oof_predictions,
    )
    return ExperimentResultV5(
        config_name=config.name,
        config_sha256=config_sha256,
        run_nonce=run_nonce,
        cache_sha256=payload.cache_sha256,
        source_script_version=(
            payload.source_script_version
        ),
        fold_assignment_sha256=(
            fold_contract.assignment_sha256
        ),
        train_order_sha256=(
            fold_contract.study_order_sha256
        ),
        train_csv_sha256=(
            fold_contract.train_csv_sha256
        ),
        weak_label_sha256=(
            report_supervision.weak_digest
        ),
        supervision_sha256s=tuple(
            supervision_sha256s
        ),
        fold_enabled_counts=tuple(
            enabled_counts
        ),
        seeds=tuple(config.seeds),
        oof_predictions=oof_predictions,
        per_target_auc=score_summary[
            "per_target"
        ],
        macro_auc=float(
            score_summary["macro_auc"]
        ),
        scored_targets=int(
            score_summary["scored_targets"]
        ),
        assignment_counts=(
            oof.assignment_counts
        ),
        fold_records=tuple(fold_records),
        runtime_seconds=float(
            time.perf_counter() - started_at
        ),
    )


def paired_stratified_bootstrap_auc_v5(
    targets,
    baseline_predictions,
    candidate_predictions,
    *,
    n_replicates=2000,
    seed=20260810,
):
    """Paired study-level bootstrap; legacy name retained for compatibility."""
    target_array = np.asarray(
        targets,
        dtype=np.float64,
    )
    baseline_array = np.asarray(
        baseline_predictions,
        dtype=np.float64,
    )
    candidate_array = np.asarray(
        candidate_predictions,
        dtype=np.float64,
    )
    if (
        target_array.ndim != 2
        or baseline_array.shape
        != target_array.shape
        or candidate_array.shape
        != target_array.shape
    ):
        raise ValueError(
            "bootstrap arrays must share one exact two-dimensional shape"
        )
    if (
        not np.all(np.isfinite(target_array))
        or not np.all(np.isfinite(baseline_array))
        or not np.all(np.isfinite(candidate_array))
    ):
        raise ValueError(
            "bootstrap arrays must be finite"
        )
    if np.any(
        (target_array != 0.0)
        & (target_array != 1.0)
    ):
        raise ValueError(
            "bootstrap targets must be exactly binary"
        )
    for name, predictions in (
        ("baseline", baseline_array),
        ("candidate", candidate_array),
    ):
        if np.any(
            (predictions < 0.0)
            | (predictions > 1.0)
        ):
            raise ValueError(
                f"{name} predictions must be probabilities"
            )
    if int(n_replicates) <= 0:
        raise ValueError(
            "n_replicates must be positive"
        )

    n_rows, n_targets = target_array.shape
    for target_index in range(n_targets):
        if np.unique(
            target_array[:, target_index]
        ).size != 2:
            raise ValueError(
                "every bootstrap target needs both classes"
            )

    rng = np.random.default_rng(int(seed))
    macro_deltas = np.full(
        int(n_replicates),
        np.nan,
        dtype=np.float64,
    )
    for replicate_index in range(
        int(n_replicates)
    ):
        sampled_rows = rng.integers(
            0,
            n_rows,
            size=n_rows,
        )
        target_deltas = []
        valid_replicate = True
        for target_index in range(n_targets):
            sampled_labels = target_array[
                sampled_rows,
                target_index,
            ]
            if np.unique(
                sampled_labels
            ).size != 2:
                valid_replicate = False
                break
            baseline_auc = roc_auc_score(
                sampled_labels,
                baseline_array[
                    sampled_rows,
                    target_index,
                ],
            )
            candidate_auc = roc_auc_score(
                sampled_labels,
                candidate_array[
                    sampled_rows,
                    target_index,
                ],
            )
            target_deltas.append(
                candidate_auc - baseline_auc
            )
        if valid_replicate:
            macro_deltas[replicate_index] = (
                np.mean(target_deltas)
            )

    valid = macro_deltas[
        np.isfinite(macro_deltas)
    ]
    if len(valid) == 0:
        raise ValueError(
            "no valid paired study bootstrap replicates"
        )
    quantiles = np.quantile(
        valid,
        [0.05, 0.20, 0.50, 0.80, 0.95],
    )
    macro_deltas.setflags(write=False)
    return MappingProxyType(
        {
            "macro_deltas": macro_deltas,
            "valid_replicates": int(
                len(valid)
            ),
            "probability_positive": float(
                np.mean(valid > 0.0)
            ),
            "p05": float(quantiles[0]),
            "p20": float(quantiles[1]),
            "median_delta": float(
                quantiles[2]
            ),
            "p80": float(quantiles[3]),
            "p95": float(quantiles[4]),
            "seed": int(seed),
        }
    )


def h0_reproduction_gate_v5(
    *,
    observed_macro_auc,
    reference_macro_auc=0.6280362353,
    tolerance=0.005,
    scored_targets,
    assignment_counts,
):
    counts = np.asarray(
        assignment_counts,
        dtype=np.int64,
    )
    reasons = []
    if not np.isfinite(observed_macro_auc):
        reasons.append(
            "observed macro AUC is non-finite"
        )
    elif abs(
        float(observed_macro_auc)
        - float(reference_macro_auc)
    ) > float(tolerance):
        reasons.append(
            "macro AUC is outside the reproduction tolerance"
        )
    if int(scored_targets) != 12:
        reasons.append(
            "not all 12 targets are scorable"
        )
    if (
        counts.shape != (58,)
        or not np.all(counts == 1)
    ):
        reasons.append(
            "OOF rows were not assigned exactly once"
        )
    return MappingProxyType(
        {
            "passed": not reasons,
            "reasons": tuple(reasons),
            "observed_macro_auc": float(
                observed_macro_auc
            ),
            "reference_macro_auc": float(
                reference_macro_auc
            ),
            "absolute_difference": float(
                abs(
                    float(observed_macro_auc)
                    - float(reference_macro_auc)
                )
            ),
            "tolerance": float(tolerance),
        }
    )


def candidate_screen_v5(
    *,
    baseline_result,
    candidate_result,
    gold_values,
    n_replicates=2000,
    seed=20260810,
):
    if (
        baseline_result.cache_sha256
        != candidate_result.cache_sha256
        or baseline_result.fold_assignment_sha256
        != candidate_result.fold_assignment_sha256
        or baseline_result.train_order_sha256
        != candidate_result.train_order_sha256
        or baseline_result.train_csv_sha256
        != candidate_result.train_csv_sha256
        or baseline_result.weak_label_sha256
        != candidate_result.weak_label_sha256
    ):
        raise ValueError(
            "candidate and baseline provenance differ"
        )
    if (
        baseline_result.scored_targets != 12
        or candidate_result.scored_targets != 12
    ):
        raise ValueError(
            "candidate screen requires 12 scorable targets"
        )
    bootstrap = paired_stratified_bootstrap_auc_v5(
        gold_values,
        baseline_result.oof_predictions,
        candidate_result.oof_predictions,
        n_replicates=n_replicates,
        seed=seed,
    )
    per_target_delta = (
        np.asarray(
            candidate_result.per_target_auc
        )
        - np.asarray(
            baseline_result.per_target_auc
        )
    )
    macro_delta = float(
        candidate_result.macro_auc
        - baseline_result.macro_auc
    )
    non_worse_targets = int(
        np.sum(per_target_delta >= -0.01)
    )
    severe_drop_targets = int(
        np.sum(per_target_delta < -0.10)
    )
    worst_target_delta = float(
        np.min(per_target_delta)
    )
    reasons = []
    if macro_delta < 0.005:
        reasons.append(
            "macro AUC gain is below 0.005"
        )
    if (
        bootstrap["probability_positive"]
        < 0.75
    ):
        reasons.append(
            "bootstrap probability is below 0.75"
        )
    if bootstrap["valid_replicates"] < 1500:
        reasons.append(
            "fewer than 1500 bootstrap replicates are valid"
        )
    if non_worse_targets < 7:
        reasons.append(
            "fewer than 7 targets are non-worse within 0.01"
        )
    if severe_drop_targets:
        reasons.append(
            "one or more target drops by more than 0.10"
        )
    per_target_delta = _readonly_array_v5(
        per_target_delta,
        dtype=np.float64,
    )
    return MappingProxyType(
        {
            "passed": not reasons,
            "reasons": tuple(reasons),
            "macro_delta": macro_delta,
            "per_target_delta": per_target_delta,
            "non_worse_targets": (
                non_worse_targets
            ),
            "severe_drop_targets": (
                severe_drop_targets
            ),
            "worst_target_delta": (
                worst_target_delta
            ),
            "bootstrap": bootstrap,
        }
    )


def experiment_summary_v5(result):
    if not isinstance(result, ExperimentResultV5):
        raise TypeError(
            "result must be ExperimentResultV5"
        )
    return {
        "config_name": result.config_name,
        "config_sha256": result.config_sha256,
        "run_nonce": result.run_nonce,
        "cache_sha256": result.cache_sha256,
        "source_script_version": (
            result.source_script_version
        ),
        "fold_assignment_sha256": (
            result.fold_assignment_sha256
        ),
        "train_order_sha256": (
            result.train_order_sha256
        ),
        "train_csv_sha256": (
            result.train_csv_sha256
        ),
        "weak_label_sha256": (
            result.weak_label_sha256
        ),
        "supervision_sha256s": list(
            result.supervision_sha256s
        ),
        "fold_enabled_counts": list(
            result.fold_enabled_counts
        ),
        "seeds": list(result.seeds),
        "per_target_auc": {
            target: float(score)
            for target, score in zip(
                TARGETS,
                result.per_target_auc,
            )
        },
        "macro_auc": float(
            result.macro_auc
        ),
        "scored_targets": int(
            result.scored_targets
        ),
        "assignment_count_min": int(
            np.min(result.assignment_counts)
        ),
        "assignment_count_max": int(
            np.max(result.assignment_counts)
        ),
        "fold_records": [
            dict(record)
            for record in result.fold_records
        ],
        "runtime_seconds": float(
            result.runtime_seconds
        ),
    }


def save_experiment_summary_v5(result, path):
    target_path = Path(path)
    if target_path.exists():
        raise FileExistsError(
            f"refusing to overwrite summary: {target_path.name}"
        )
    target_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = target_path.with_name(
        f".{target_path.name}.{uuid.uuid4().hex}.tmp"
    )
    payload = experiment_summary_v5(result)
    try:
        with temporary_path.open(
            "x",
            encoding="utf-8",
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary_path,
            target_path,
        )
    finally:
        temporary_path.unlink(
            missing_ok=True
        )
    return _file_sha256_v5(target_path)


    

# %%

# ============================================================
# Version 5 Task 10 RED/GREEN — OOF experiment orchestration
# ============================================================
from dataclasses import replace

_task10_calls = []

def _fake_fold_trainer_v5(**kwargs):
    validation_indices = np.asarray(
        kwargs["validation_indices"],
        dtype=np.int64,
    )
    train_indices = np.asarray(
        kwargs["train_indices"],
        dtype=np.int64,
    )
    validation_targets = np.asarray(
        kwargs["validation_targets"],
        dtype=np.float32,
    )
    validation_weights = np.asarray(
        kwargs["validation_weights"],
        dtype=np.float32,
    )
    training_weights = np.asarray(
        kwargs["training_weights"],
        dtype=np.float32,
    )
    truth = validation_targets[validation_indices]
    weights = validation_weights[validation_indices]
    assert np.all(np.isfinite(truth))
    assert np.all(weights == 1.0)
    predictions = (
        0.10 + 0.80 * truth
    ).astype(np.float64)
    predictions.setflags(write=False)
    readonly_validation = validation_indices.copy()
    readonly_validation.setflags(write=False)
    identity = kwargs["checkpoint_identity"]
    _task10_calls.append(
        MappingProxyType(
            {
                "validation_indices": readonly_validation,
                "train_indices": train_indices.copy(),
                "heldout_training_weight": float(
                    training_weights[validation_indices].sum()
                ),
                "seed": int(kwargs["seed"]),
                "identity": identity,
            }
        )
    )
    return FoldRunResult(
        predictions=predictions,
        validation_indices=readonly_validation,
        state_dict=MappingProxyType(
            {"synthetic": torch.ones(1)}
        ),
        best_epoch=1,
        best_score=1.0,
        macro_auc=1.0,
        scored_targets=truth.shape[1],
        validation_loss=0.1,
        epochs_ran=1,
        train_rows=len(train_indices),
        validation_rows=len(validation_indices),
        runtime_seconds=0.01,
        checkpoint_path=str(kwargs["checkpoint_path"]),
        checkpoint_sha256="e" * 64,
    )

_submission_path_before = Path(
    "/kaggle/working/submission.csv"
)
_submission_existed_before = _submission_path_before.exists()
_verified_runner_payload = load_verified_training_cache(
    TRAIN_CACHE_CONTRACT
)
_h0_contract_result = run_oof_experiment_v5(
    config=get_experiment("v3_repro"),
    payload=_verified_runner_payload,
    fold_contract=FOLD_CONTRACT,
    report_supervision=REPORT_SUPERVISION,
    model_factory=TargetAttentionMILV3,
    device="cpu",
    checkpoint_root=Path(
        "/kaggle/working/v5_task10_fake"
    ),
    run_nonce="task10-h0-contract",
    batch_size=32,
    trainer=_fake_fold_trainer_v5,
)
assert len(_task10_calls) == 5
assert _h0_contract_result.oof_predictions.shape == (58, 12)
assert not _h0_contract_result.oof_predictions.flags.writeable
assert not _h0_contract_result.per_target_auc.flags.writeable
assert not _h0_contract_result.assignment_counts.flags.writeable
assert np.all(
    _h0_contract_result.assignment_counts == 1
)
assert _h0_contract_result.scored_targets == 12
assert _h0_contract_result.macro_auc == 1.0
np.testing.assert_allclose(
    _h0_contract_result.per_target_auc,
    np.ones(12),
)
assert _h0_contract_result.fold_enabled_counts == (
    12,
    12,
    12,
    12,
    12,
)
assert (
    _h0_contract_result.cache_sha256
    == TRAIN_CACHE_CONTRACT.expected_sha256
)
assert (
    _h0_contract_result.fold_assignment_sha256
    == FOLD_CONTRACT.assignment_sha256
)
assert len(set(_h0_contract_result.supervision_sha256s)) == 5
for _fold_index, _call in enumerate(_task10_calls):
    _train_relative, _valid_relative = (
        FOLD_CONTRACT.folds[_fold_index]
    )
    _expected_valid_global = FOLD_CONTRACT.gold_positions[
        _valid_relative
    ]
    np.testing.assert_array_equal(
        _call["validation_indices"],
        _expected_valid_global,
    )
    assert len(_call["train_indices"]) == (
        len(TRAIN_DF) - len(_expected_valid_global)
    )
    assert not np.intersect1d(
        _call["train_indices"],
        _expected_valid_global,
    ).size
    assert _call["heldout_training_weight"] == 0.0
    assert _call["seed"] == 20260808 + _fold_index
    assert _call["identity"].experiment_id == "v3_repro"
    assert (
        _call["identity"].fold_index
        == _fold_index
    )
    assert (
        _call["identity"].cache_sha256
        == TRAIN_CACHE_CONTRACT.expected_sha256
    )

_task10_calls.clear()
_h1_contract_result = run_oof_experiment_v5(
    config=get_experiment("fold_gated"),
    payload=_verified_runner_payload,
    fold_contract=FOLD_CONTRACT,
    report_supervision=REPORT_SUPERVISION,
    model_factory=TargetAttentionMILV3,
    device="cpu",
    checkpoint_root=Path(
        "/kaggle/working/v5_task10_fake"
    ),
    run_nonce="task10-h1-contract",
    batch_size=32,
    trainer=_fake_fold_trainer_v5,
)
assert len(_task10_calls) == 5
assert _h1_contract_result.fold_enabled_counts == (
    8,
    8,
    8,
    8,
    8,
)
assert (
    _h1_contract_result.config_sha256
    != _h0_contract_result.config_sha256
)
assert (
    _h1_contract_result.fold_assignment_sha256
    == _h0_contract_result.fold_assignment_sha256
)
assert (
    _h1_contract_result.cache_sha256
    == _h0_contract_result.cache_sha256
)
assert (
    _h1_contract_result.supervision_sha256s
    != _h0_contract_result.supervision_sha256s
)

_bootstrap_targets = np.tile(
    np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    (6, 1),
)
_bootstrap_baseline = np.full(
    _bootstrap_targets.shape,
    0.5,
    dtype=np.float64,
)
_bootstrap_candidate = (
    0.1 + 0.8 * _bootstrap_targets
)
_bootstrap_a = paired_stratified_bootstrap_auc_v5(
    _bootstrap_targets,
    _bootstrap_baseline,
    _bootstrap_candidate,
    n_replicates=200,
    seed=20260810,
)
_bootstrap_b = paired_stratified_bootstrap_auc_v5(
    _bootstrap_targets,
    _bootstrap_baseline,
    _bootstrap_candidate,
    n_replicates=200,
    seed=20260810,
)
assert _bootstrap_a["valid_replicates"] == 200
assert _bootstrap_a["probability_positive"] == 1.0
assert _bootstrap_a["median_delta"] == 0.5
assert not _bootstrap_a["macro_deltas"].flags.writeable
np.testing.assert_array_equal(
    _bootstrap_a["macro_deltas"],
    _bootstrap_b["macro_deltas"],
)


_study_rows = np.arange(12, dtype=np.int64)
_study_targets = np.column_stack(
    (
        (_study_rows % 2).astype(np.float64),
        ((_study_rows // 2) % 2).astype(np.float64),
        (_study_rows == 0).astype(np.float64),
    )
)
_study_rng = np.random.default_rng(707)
_study_baseline = _study_rng.uniform(
    0.05,
    0.95,
    size=_study_targets.shape,
)
_study_candidate = np.clip(
    _study_baseline
    + (2.0 * _study_targets - 1.0) * 0.12,
    0.0,
    1.0,
)

def _manual_paired_study_bootstrap(
    targets,
    baseline,
    candidate,
    *,
    n_replicates,
    seed,
):
    rng = np.random.default_rng(seed)
    deltas = np.full(
        n_replicates,
        np.nan,
        dtype=np.float64,
    )
    for replicate_index in range(n_replicates):
        sampled_rows = rng.integers(
            0,
            len(targets),
            size=len(targets),
        )
        target_deltas = []
        valid = True
        for target_index in range(targets.shape[1]):
            labels = targets[
                sampled_rows,
                target_index,
            ]
            if len(np.unique(labels)) != 2:
                valid = False
                break
            target_deltas.append(
                roc_auc_score(
                    labels,
                    candidate[
                        sampled_rows,
                        target_index,
                    ],
                )
                - roc_auc_score(
                    labels,
                    baseline[
                        sampled_rows,
                        target_index,
                    ],
                )
            )
        if valid:
            deltas[replicate_index] = np.mean(
                target_deltas
            )
    return deltas

_study_bootstrap = (
    paired_stratified_bootstrap_auc_v5(
        _study_targets,
        _study_baseline,
        _study_candidate,
        n_replicates=120,
        seed=321,
    )
)
_manual_study_deltas = (
    _manual_paired_study_bootstrap(
        _study_targets,
        _study_baseline,
        _study_candidate,
        n_replicates=120,
        seed=321,
    )
)
np.testing.assert_allclose(
    _study_bootstrap["macro_deltas"],
    _manual_study_deltas,
    rtol=0.0,
    atol=1e-12,
    equal_nan=True,
)
assert _study_bootstrap["valid_replicates"] == int(
    np.isfinite(_manual_study_deltas).sum()
)
assert _study_bootstrap["valid_replicates"] < 120

_h0_gate_pass = h0_reproduction_gate_v5(
    observed_macro_auc=0.6280362353,
    reference_macro_auc=0.6280362353,
    tolerance=0.005,
    scored_targets=12,
    assignment_counts=np.ones(58, dtype=np.int64),
)
assert _h0_gate_pass["passed"]
_h0_gate_fail = h0_reproduction_gate_v5(
    observed_macro_auc=0.64,
    reference_macro_auc=0.6280362353,
    tolerance=0.005,
    scored_targets=12,
    assignment_counts=np.ones(58, dtype=np.int64),
)
assert not _h0_gate_fail["passed"]


_drop_predictions = (
    _h0_contract_result.oof_predictions.copy()
)
_drop_predictions[:, -1] = (
    1.0 - _drop_predictions[:, -1]
)
_drop_predictions.setflags(write=False)
_drop_summary = auc_summary_v5(
    FOLD_CONTRACT.gold_values,
    _drop_predictions,
)
_drop_candidate = replace(
    _h1_contract_result,
    oof_predictions=_drop_predictions,
    per_target_auc=_drop_summary["per_target"],
    macro_auc=_drop_summary["macro_auc"],
    scored_targets=_drop_summary[
        "scored_targets"
    ],
)
_drop_screen = candidate_screen_v5(
    baseline_result=_h0_contract_result,
    candidate_result=_drop_candidate,
    gold_values=FOLD_CONTRACT.gold_values,
    n_replicates=120,
    seed=321,
)
assert not _drop_screen["passed"]
assert _drop_screen["severe_drop_targets"] == 1
assert _drop_screen["worst_target_delta"] < -0.10
assert any(
    "target drops by more than 0.10" in reason
    for reason in _drop_screen["reasons"]
)

assert (
    _submission_path_before.exists()
    == _submission_existed_before
)
print(
    "V5 EXPERIMENT RUNNER CONTRACT PASSED: "
    "folds=5, oof=(58, 12), "
    "H0_enabled=12, H1_enabled=8, "
    "bootstrap_deterministic=True"
)


    

# %%

# ============================================================
# Version 5 controlled experiment execution (saved idle)
# ============================================================
RUN_EXPERIMENT_REQUEST = None
RUN_EXPERIMENT_NONCE = None

if "V5_EXPERIMENT_RESULTS" not in globals():
    V5_EXPERIMENT_RESULTS = {}

if RUN_EXPERIMENT_REQUEST is None:
    print(
        "V5 EXPERIMENT CONTROL IDLE: "
        "set an explicit registered request and execute this cell manually"
    )
else:
    import gc

    if RUN_EXPERIMENT_REQUEST not in (
        "v3_repro",
        "fold_gated",
        "gold_weight_2",
    ):
        raise ValueError(
            "only reviewed H0/H1/H2 experiments are enabled here"
        )
    if (
        not isinstance(
            RUN_EXPERIMENT_NONCE,
            str,
        )
        or not RUN_EXPERIMENT_NONCE.strip()
    ):
        raise ValueError(
            "an explicit unique run nonce is required"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "reviewed head experiment requires an active CUDA session"
        )

    _requested_config = get_experiment(
        RUN_EXPERIMENT_REQUEST
    )
    _requested_payload = (
        load_verified_training_cache(
            TRAIN_CACHE_CONTRACT
        )
    )
    print(
        "V5 EXPERIMENT START: "
        f"name={_requested_config.name}, "
        f"device={torch.cuda.get_device_name(0)}, "
        f"folds={FOLD_CONTRACT.n_folds}, "
        f"seeds={_requested_config.seeds}, "
        f"epochs={_requested_config.epochs}, "
        f"patience={_requested_config.patience}"
    )
    _requested_result = run_oof_experiment_v5(
        config=_requested_config,
        payload=_requested_payload,
        fold_contract=FOLD_CONTRACT,
        report_supervision=REPORT_SUPERVISION,
        model_factory=TargetAttentionMILV3,
        device="cuda",
        checkpoint_root=Path(
            "/kaggle/working/v5_checkpoints"
        ),
        run_nonce=RUN_EXPERIMENT_NONCE,
        batch_size=32,
        grad_clip=1.0,
    )
    _summary_path = Path(
        "/kaggle/working"
    ) / (
        f"v5_metrics_{_requested_config.name}"
        f"__{RUN_EXPERIMENT_NONCE}.json"
    )
    _summary_sha256 = save_experiment_summary_v5(
        _requested_result,
        _summary_path,
    )
    V5_EXPERIMENT_RESULTS[
        _requested_config.name
    ] = _requested_result

    print(
        "V5 EXPERIMENT COMPLETE: "
        f"name={_requested_result.config_name}, "
        f"macro_auc={_requested_result.macro_auc:.10f}, "
        f"scored_targets={_requested_result.scored_targets}, "
        f"runtime_seconds={_requested_result.runtime_seconds:.1f}, "
        f"summary_sha256={_summary_sha256}"
    )
    print(
        "V5 PER-TARGET AUC:",
        {
            target: round(float(score), 6)
            for target, score in zip(
                TARGETS,
                _requested_result.per_target_auc,
            )
        },
    )
    print(
        "V5 FOLD AGGREGATES:",
        [
            {
                key: value
                for key, value in dict(record).items()
                if key != "checkpoint_sha256"
            }
            for record in _requested_result.fold_records
        ],
    )
    if _requested_config.name == "v3_repro":
        _h0_gate = h0_reproduction_gate_v5(
            observed_macro_auc=(
                _requested_result.macro_auc
            ),
            reference_macro_auc=0.6280362353,
            tolerance=0.005,
            scored_targets=(
                _requested_result.scored_targets
            ),
            assignment_counts=(
                _requested_result.assignment_counts
            ),
        )
        print(
            "V5 H0 REPRODUCTION GATE:",
            dict(_h0_gate),
        )
    elif (
        _requested_config.name == "fold_gated"
        and "v3_repro" in V5_EXPERIMENT_RESULTS
    ):
        _candidate_screen = candidate_screen_v5(
            baseline_result=(
                V5_EXPERIMENT_RESULTS[
                    "v3_repro"
                ]
            ),
            candidate_result=_requested_result,
            gold_values=FOLD_CONTRACT.gold_values,
            n_replicates=2000,
            seed=20260810,
        )
        print(
            "V5 H1 SCREEN:",
            {
                "passed": _candidate_screen[
                    "passed"
                ],
                "reasons": _candidate_screen[
                    "reasons"
                ],
                "macro_delta": (
                    _candidate_screen[
                        "macro_delta"
                    ]
                ),
                "non_worse_targets": (
                    _candidate_screen[
                        "non_worse_targets"
                    ]
                ),
                "severe_drop_targets": (
                    _candidate_screen[
                        "severe_drop_targets"
                    ]
                ),
                "worst_target_delta": (
                    _candidate_screen[
                        "worst_target_delta"
                    ]
                ),
                "bootstrap": {
                    key: value
                    for key, value in dict(
                        _candidate_screen[
                            "bootstrap"
                        ]
                    ).items()
                    if key != "macro_deltas"
                },
            },
        )

    del _requested_payload
    gc.collect()
    torch.cuda.empty_cache()


    

# %%

# ============================================================
# Version 5 verified experiment evidence — aggregate only
# ============================================================
from types import MappingProxyType

_H0_TARGET_AUC = MappingProxyType(
    {
        "ACL": 0.579657,
        "MCL": 0.539683,
        "Medial Meniscus": 0.616587,
        "Lateral Meniscus": 0.731677,
        "Medial OA": 0.644961,
        "Lateral OA": 0.736944,
        "PF OA": 0.491634,
        "Effusion": 0.706832,
        "Synovitis": 0.561529,
        "Baker's": 0.711957,
        "Contusion": 0.724696,
        "Fracture": 0.490278,
    }
)
_H1_TARGET_AUC = MappingProxyType(
    {
        "ACL": 0.556373,
        "MCL": 0.603175,
        "Medial Meniscus": 0.621394,
        "Lateral Meniscus": 0.746584,
        "Medial OA": 0.480620,
        "Lateral OA": 0.586074,
        "PF OA": 0.572716,
        "Effusion": 0.781366,
        "Synovitis": 0.673835,
        "Baker's": 0.784420,
        "Contusion": 0.747638,
        "Fracture": 0.583333,
    }
)
_H2_TARGET_AUC = MappingProxyType(
    {
        "ACL": 0.537990,
        "MCL": 0.537415,
        "Medial Meniscus": 0.590144,
        "Lateral Meniscus": 0.686957,
        "Medial OA": 0.482171,
        "Lateral OA": 0.493230,
        "PF OA": 0.558559,
        "Effusion": 0.819876,
        "Synovitis": 0.594982,
        "Baker's": 0.817029,
        "Contusion": 0.798920,
        "Fracture": 0.494444,
    }
)

VERIFIED_EXPERIMENT_EVIDENCE = MappingProxyType(
    {
        "h0_v3_repro": MappingProxyType(
            {
                "macro_auc": 0.6280362353,
                "scored_targets": 12,
                "oof_rows": 58,
                "assignment_min": 1,
                "assignment_max": 1,
                "runtime_seconds": 33.0,
                "summary_sha256": (
                    "ca3aaa05246b498bc9024e1257c10dbb"
                    "e3d6aa1a7492348f1fb315448bd77e81"
                ),
                "reproduction_gate_passed": True,
                "per_target_auc": _H0_TARGET_AUC,
            }
        ),
        "h1_fold_gated": MappingProxyType(
            {
                "macro_auc": 0.6447939998,
                "macro_delta_vs_h0": 0.0167577645,
                "scored_targets": 12,
                "runtime_seconds": 34.6,
                "summary_sha256": (
                    "e6cce4b7a64baeea13b65c3cc2d1ba2"
                    "ea80d83ef01acd7f6d9785cf602da3b03"
                ),
                "screen_gate_passed": False,
                "confirm_gate_passed": False,
                "screen_blockers": (
                    "bootstrap_probability_below_0.75",
                    "two_target_drops_exceed_0.10_vs_h0",
                ),
                "confirm_blockers": (
                    "macro_auc_below_0.645_by_0.000206",
                    "bootstrap_probability_below_0.75",
                    "two_target_drops_exceed_0.10_vs_h0",
                ),
                "non_worse_targets_within_0.01": 9,
                "severe_drop_targets": 2,
                "worst_target_delta": -0.1643410853,
                "bootstrap_method": "paired_study_level",
                "bootstrap_replicates": 2000,
                "bootstrap_valid_replicates": 2000,
                "bootstrap_probability_positive": 0.7485,
                "bootstrap_p05": -0.0248013415,
                "bootstrap_p20": -0.0047184029,
                "bootstrap_median_delta": 0.0155709819,
                "bootstrap_p80": 0.0361153193,
                "bootstrap_p95": 0.0573023905,
                "bootstrap_seed": 20260810,
                "per_target_auc": _H1_TARGET_AUC,
            }
        ),
        "h2_gold_weight_2": MappingProxyType(
            {
                "macro_auc": 0.6176430711,
                "macro_delta_vs_h1": -0.0271509287,
                "scored_targets": 12,
                "runtime_seconds": 32.3,
                "summary_sha256": (
                    "0458de107b1c670d6a1ff34bfb13fe9"
                    "63037944fde49b5ef1bb2dce7f21a8f94"
                ),
                "promotion_gate_passed": False,
                "medial_oa_delta_vs_h1": 0.001551,
                "lateral_oa_delta_vs_h1": -0.092844,
                "per_target_auc": _H2_TARGET_AUC,
            }
        ),
        "decision": MappingProxyType(
            {
                "retained_confirmed_candidate": "v3_repro",
                "promising_unconfirmed_candidate": None,
                "h3_not_run_reason": (
                    "H2 failed macro and OA-recovery continuation gates"
                ),
                "h4_not_run_reason": (
                    "no single-seed candidate passed confirm gates"
                ),
                "competition_submission_created": False,
                "gpu_minutes_observed": 10,
                "gpu_stopped_after_experiments": True,
                "accelerator_reset_to_none": True,
                "internet_off": True,
            }
        ),
    }
)

assert tuple(_H0_TARGET_AUC) == TARGETS
assert tuple(_H1_TARGET_AUC) == TARGETS
assert tuple(_H2_TARGET_AUC) == TARGETS
assert np.isclose(
    VERIFIED_EXPERIMENT_EVIDENCE[
        "h1_fold_gated"
    ]["macro_delta_vs_h0"],
    VERIFIED_EXPERIMENT_EVIDENCE[
        "h1_fold_gated"
    ]["macro_auc"]
    - VERIFIED_EXPERIMENT_EVIDENCE[
        "h0_v3_repro"
    ]["macro_auc"],
    atol=1e-10,
)
assert not VERIFIED_EXPERIMENT_EVIDENCE[
    "h1_fold_gated"
]["screen_gate_passed"]
assert VERIFIED_EXPERIMENT_EVIDENCE[
    "h1_fold_gated"
]["bootstrap_method"] == "paired_study_level"
assert VERIFIED_EXPERIMENT_EVIDENCE[
    "h1_fold_gated"
]["bootstrap_valid_replicates"] == 2000
assert np.isclose(
    VERIFIED_EXPERIMENT_EVIDENCE[
        "h1_fold_gated"
    ]["bootstrap_probability_positive"],
    0.7485,
)
assert VERIFIED_EXPERIMENT_EVIDENCE[
    "h1_fold_gated"
]["severe_drop_targets"] == 2
assert not Path(
    "/kaggle/working/submission.csv"
).exists()

print(
    "V5 VERIFIED EXPERIMENT EVIDENCE PASSED: "
    "H0=0.628036, H1=0.644794 (screen failed), "
    "H2=0.617643 (rejected), "
    "retained=v3_repro, submission_created=False"
)


    

# %% [markdown]
# # RSNA Knee V5 Cached Head Lab
#
# ## Purpose
#
# This private notebook improves the verified Version 3 model through leakage-safe out-of-fold experiments. It is a research competition workflow, not a clinical diagnostic system.
#
# ## Hard boundaries
#
# - OOF experiments only: this notebook must not create or submit `submission.csv`.
# - All behavior changes are written, tested, and fixed in this Kaggle browser notebook.
# - The exact Version 3 feature cache is reused read-only; no DICOM or DINO encoding occurs here.
# - Every change is tested one factor at a time against identical folds and official labels.
# - Reports, identifiers, predictions, cached features, and checkpoints are never printed or copied to Git.
# - Internet stays off. Accelerator stays None unless a measured promoted head run needs T4, and T4 is stopped immediately afterward.
#
# ## Promotion rule
#
# A candidate must pass all ordering, fold-local supervision, held-out immutability, cache provenance, and exact OOF contracts before its score is considered.
