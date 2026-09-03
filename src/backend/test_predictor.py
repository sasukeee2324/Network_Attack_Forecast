from pathlib import Path

import numpy as np

from .config import (
    DATA_DIR,
    HISTORY_MINUTES,
    FEATURE_COUNT,
)
from .feature_pipeline import (
    generate_enhanced_features,
    create_model_window,
)
from .predictor import (
    NetworkAttackPredictor,
)


# ============================================================
# INPUT DATA
# ============================================================

INPUT_FILE = (
    DATA_DIR
    / "temporal"
    / "network_states.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("NETWORK ATTACK FORECASTING - INFERENCE TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nNetwork-state file not found:\n"
            f"{INPUT_FILE}"
        )

    print(
        f"\nInput file:\n{INPUT_FILE}"
    )

    # --------------------------------------------------------
    # LOAD NETWORK STATES
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Network states loaded: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # GENERATE ENHANCED FEATURES
    # --------------------------------------------------------

    print(
        "\nGenerating enhanced temporal features..."
    )

    enhanced_df = (
        generate_enhanced_features(df)
    )

    print(
        f"Enhanced dataset shape: "
        f"{enhanced_df.shape}"
    )

    # --------------------------------------------------------
    # SELECT ONE SOURCE
    # --------------------------------------------------------

    sources = (
        enhanced_df["Source_File"]
        .dropna()
        .unique()
    )

    if len(sources) == 0:
        raise RuntimeError(
            "No Source_File values found."
        )

    source = sources[0]

    source_df = (
        enhanced_df[
            enhanced_df["Source_File"] == source
        ]
        .sort_values("Minute")
        .reset_index(drop=True)
    )

    print(
        f"\nUsing source:\n{source}"
    )

    print(
        f"Available states: "
        f"{len(source_df)}"
    )

    # --------------------------------------------------------
    # CREATE 10-MINUTE WINDOW
    # --------------------------------------------------------

    if len(source_df) < HISTORY_MINUTES:
        raise RuntimeError(
            "Not enough states for "
            "a 10-minute window."
        )

    window = create_model_window(
        source_df
    )

    print(
        f"\nModel window shape: "
        f"{window.shape}"
    )

    if window.shape != (
        HISTORY_MINUTES,
        FEATURE_COUNT,
    ):
        raise RuntimeError(
            f"Unexpected window shape: "
            f"{window.shape}"
        )

    print(
        "Window validation: PASSED"
    )

    # --------------------------------------------------------
    # LOAD PREDICTOR
    # --------------------------------------------------------

    print(
        "\nLoading forecasting model..."
    )

    predictor = (
        NetworkAttackPredictor()
    )

    print(
        "Model loading: PASSED"
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    print(
        "\nGenerating forecast..."
    )

    result = predictor.predict(
        window
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "FORECAST RESULT"
    )

    print(
        "=" * 80
    )

    for key, value in result.items():

        if isinstance(value, float):

            print(
                f"{key}: "
                f"{value:.6f}"
            )

        else:

            print(
                f"{key}: "
                f"{value}"
            )

    print(
        "\n" + "=" * 80
    )

    print(
        "INFERENCE TEST PASSED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    import pandas as pd

    main()