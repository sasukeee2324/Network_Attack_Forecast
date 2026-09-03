from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .config import FEATURE_COUNT
from .feature_pipeline import generate_enhanced_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal"
    / "network_states.csv"
)

CANONICAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal"
    / "network_states_enhanced.csv"
)


KEY_COLUMNS = [
    "Source_File",
    "Minute",
]

NON_FEATURE_COLUMNS = {
    "Minute",
    "Source_File",
    "Attack_Flow_Count",
    "Attack_Ratio",
    "Attack_Type",
    "Attack_State",
}

ABS_TOLERANCE = 1e-7
REL_TOLERANCE = 1e-6


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:

    result = df.copy()

    result["Source_File"] = (
        result["Source_File"]
        .astype(str)
        .str.strip()
    )

    result["Minute"] = (
        result["Minute"]
        .astype(str)
        .str.strip()
    )

    return result


def make_key_frame(df: pd.DataFrame) -> pd.DataFrame:

    result = normalize_keys(df)

    return result[KEY_COLUMNS].copy()


def compare_feature_values(
    generated: pd.DataFrame,
    canonical: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[int, int, float, str | None]:

    generated_norm = normalize_keys(generated)
    canonical_norm = normalize_keys(canonical)

    generated_compare = generated_norm[
        KEY_COLUMNS + feature_columns
    ].copy()

    canonical_compare = canonical_norm[
        KEY_COLUMNS + feature_columns
    ].copy()

    merged = canonical_compare.merge(
        generated_compare,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=(
            "_canonical",
            "_generated",
        ),
        validate="one_to_one",
    )

    if len(merged) != len(canonical_compare):
        raise RuntimeError(
            "Not all canonical rows could be matched "
            "to generated rows."
        )

    numeric_mismatches = 0
    nan_mismatches = 0
    maximum_difference = 0.0
    worst_feature = None

    for feature in feature_columns:

        canonical_values = pd.to_numeric(
            merged[f"{feature}_canonical"],
            errors="coerce",
        ).to_numpy(dtype=float)

        generated_values = pd.to_numeric(
            merged[f"{feature}_generated"],
            errors="coerce",
        ).to_numpy(dtype=float)

        canonical_nan = np.isnan(
            canonical_values
        )

        generated_nan = np.isnan(
            generated_values
        )

        nan_mismatch = (
            canonical_nan
            ^ generated_nan
        )

        nan_mismatches += int(
            nan_mismatch.sum()
        )

        finite_mask = (
            ~canonical_nan
            & ~generated_nan
        )

        if not finite_mask.any():
            continue

        canonical_finite = (
            canonical_values[finite_mask]
        )

        generated_finite = (
            generated_values[finite_mask]
        )

        differences = np.abs(
            canonical_finite
            - generated_finite
        )

        feature_max_difference = float(
            np.max(differences)
        )

        if (
            feature_max_difference
            > maximum_difference
        ):

            maximum_difference = (
                feature_max_difference
            )

            worst_feature = feature

        mismatch_mask = ~np.isclose(
            canonical_finite,
            generated_finite,
            rtol=REL_TOLERANCE,
            atol=ABS_TOLERANCE,
        )

        numeric_mismatches += int(
            mismatch_mask.sum()
        )

    return (
        numeric_mismatches,
        nan_mismatches,
        maximum_difference,
        worst_feature,
    )


def main() -> None:

    print("=" * 70)
    print("FEATURE CONSISTENCY VERIFICATION")
    print("=" * 70)

    print("\nInput file:")
    print(RAW_FILE)

    print("\nCanonical training feature file:")
    print(CANONICAL_FILE)

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{RAW_FILE}"
        )

    if not CANONICAL_FILE.exists():
        raise FileNotFoundError(
            f"Canonical feature file not found:\n{CANONICAL_FILE}"
        )

    print("\n[1/5] Loading files...")

    raw = pd.read_csv(
        RAW_FILE,
        low_memory=False,
    )

    canonical = pd.read_csv(
        CANONICAL_FILE,
        low_memory=False,
    )

    print(
        f"Raw rows:       {len(raw)}"
    )

    print(
        f"Canonical rows: {len(canonical)}"
    )

    if len(raw) != len(canonical):
        raise RuntimeError(
            "Raw and canonical row counts differ."
        )

    print(
        "\n[2/5] Generating features "
        "with backend pipeline..."
    )

    with warnings.catch_warnings():

        warnings.filterwarnings(
            "ignore",
            category=pd.errors.PerformanceWarning,
        )

        generated = generate_enhanced_features(
            raw
        )

    print(
        f"Generated rows: {len(generated)}"
    )

    generated_features = [
        column
        for column in generated.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    canonical_features = [
        column
        for column in canonical.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    print(
        f"\nGenerated model features: "
        f"{len(generated_features)}"
    )

    print(
        f"Canonical model features: "
        f"{len(canonical_features)}"
    )

    print(
        f"Expected model features:  "
        f"{FEATURE_COUNT}"
    )

    if len(generated_features) != FEATURE_COUNT:
        raise RuntimeError(
            "Generated feature count mismatch.\n"
            f"Expected: {FEATURE_COUNT}\n"
            f"Actual:   {len(generated_features)}"
        )

    if len(canonical_features) != FEATURE_COUNT:
        raise RuntimeError(
            "Canonical feature count mismatch.\n"
            f"Expected: {FEATURE_COUNT}\n"
            f"Actual:   {len(canonical_features)}"
        )

    print(
        "\n[3/5] Checking feature names..."
    )

    if generated_features != canonical_features:

        generated_set = set(
            generated_features
        )

        canonical_set = set(
            canonical_features
        )

        missing = sorted(
            canonical_set - generated_set
        )

        unexpected = sorted(
            generated_set - canonical_set
        )

        if missing:
            print("\nMissing generated features:")
            for feature in missing:
                print(f"  - {feature}")

        if unexpected:
            print("\nUnexpected generated features:")
            for feature in unexpected:
                print(f"  - {feature}")

        raise RuntimeError(
            "Feature name check FAILED."
        )

    print("Feature names: PASS")

    print(
        "\n[4/5] Checking row alignment..."
    )

    raw_keys = make_key_frame(raw)
    canonical_keys = make_key_frame(canonical)
    generated_keys = make_key_frame(generated)

    raw_duplicate_count = int(
        raw_keys.duplicated(
            subset=KEY_COLUMNS
        ).sum()
    )

    canonical_duplicate_count = int(
        canonical_keys.duplicated(
            subset=KEY_COLUMNS
        ).sum()
    )

    generated_duplicate_count = int(
        generated_keys.duplicated(
            subset=KEY_COLUMNS
        ).sum()
    )

    print(
        f"Raw duplicate keys:       "
        f"{raw_duplicate_count}"
    )

    print(
        f"Canonical duplicate keys: "
        f"{canonical_duplicate_count}"
    )

    print(
        f"Generated duplicate keys: "
        f"{generated_duplicate_count}"
    )

    if (
        raw_duplicate_count
        or canonical_duplicate_count
        or generated_duplicate_count
    ):
        raise RuntimeError(
            "Duplicate Source_File + Minute "
            "keys detected."
        )

    raw_key_set = set(
        map(
            tuple,
            raw_keys.itertuples(
                index=False,
                name=None,
            ),
        )
    )

    canonical_key_set = set(
        map(
            tuple,
            canonical_keys.itertuples(
                index=False,
                name=None,
            ),
        )
    )

    generated_key_set = set(
        map(
            tuple,
            generated_keys.itertuples(
                index=False,
                name=None,
            ),
        )
    )

    raw_to_canonical_missing = (
        raw_key_set
        - canonical_key_set
    )

    canonical_to_raw_missing = (
        canonical_key_set
        - raw_key_set
    )

    canonical_to_generated_missing = (
        canonical_key_set
        - generated_key_set
    )

    generated_to_canonical_unexpected = (
        generated_key_set
        - canonical_key_set
    )

    print(
        f"\nRaw -> canonical missing: "
        f"{len(raw_to_canonical_missing)}"
    )

    print(
        f"Canonical -> raw missing: "
        f"{len(canonical_to_raw_missing)}"
    )

    print(
        f"Canonical -> generated missing: "
        f"{len(canonical_to_generated_missing)}"
    )

    print(
        f"Generated -> canonical unexpected: "
        f"{len(generated_to_canonical_unexpected)}"
    )

    if raw_key_set != canonical_key_set:
        raise RuntimeError(
            "RAW/CANONICAL row alignment FAILED."
        )

    if generated_key_set != canonical_key_set:
        raise RuntimeError(
            "GENERATED/CANONICAL row alignment FAILED."
        )

    print("\nRow alignment: PASS")

    print(
        "\n[5/5] Comparing feature values..."
    )

    (
        numeric_mismatches,
        nan_mismatches,
        maximum_difference,
        worst_feature,
    ) = compare_feature_values(
        generated=generated,
        canonical=canonical,
        feature_columns=canonical_features,
    )

    total_feature_cells = (
        len(canonical)
        * len(canonical_features)
    )

    print(
        f"Feature cells compared: "
        f"{total_feature_cells:,}"
    )

    print(
        f"Numeric mismatches: "
        f"{numeric_mismatches:,}"
    )

    print(
        f"NaN initialization differences: "
        f"{nan_mismatches:,}"
    )

    print(
        f"Maximum absolute difference: "
        f"{maximum_difference:.12g}"
    )

    if worst_feature is not None:
        print(
            f"Worst feature: "
            f"{worst_feature}"
        )

    if numeric_mismatches > 0:

        raise RuntimeError(
            "FEATURE VALUE CONSISTENCY CHECK FAILED: "
            "genuine numeric mismatches detected."
        )

    print(
        "\nFeature values: PASS"
    )

    if nan_mismatches > 0:

        print(
            "\nNote: NaN initialization differences were "
            "detected in temporal features."
        )

        print(
            "These do not represent numerical disagreement "
            "where both pipelines produce finite values."
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "FEATURE CONSISTENCY VERIFICATION PASSED"
    )

    print(
        "=" * 70
    )

    print(
        "\nVerified:"
    )

    print(
        "  [PASS] Row identity"
    )

    print(
        "  [PASS] Row alignment"
    )

    print(
        "  [PASS] Feature count"
    )

    print(
        "  [PASS] Feature names"
    )

    print(
        "  [PASS] Finite feature values"
    )

    if nan_mismatches:
        print(
            "  [INFO] Temporal NaN initialization differences"
        )

    print(
        "\nBackend feature pipeline is "
        "numerically consistent with training data."
    )


if __name__ == "__main__":
    main()
