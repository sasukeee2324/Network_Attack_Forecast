from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path(
    "data/processed/enhanced_sequences"
)

METADATA_FILE = (
    DATA_DIR / "metadata_enhanced.csv"
)

EMBARGO_MINUTES = 25

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.10


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ENHANCED DATASET SPLIT DIAGNOSTIC")
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing metadata file:\n{METADATA_FILE}"
        )

    metadata = pd.read_csv(
        METADATA_FILE,
        parse_dates=["Forecast_Start"]
    )

    metadata = metadata.sort_values(
        ["Source_File", "Forecast_Start"]
    ).reset_index(drop=True)

    print(
        f"\nTotal enhanced sequences: "
        f"{len(metadata):,}"
    )

    print(
        f"Embargo: "
        f"{EMBARGO_MINUTES} minutes"
    )

    # --------------------------------------------------------
    # GLOBAL DISTRIBUTION
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("GLOBAL TARGET DISTRIBUTION")
    print("=" * 80)

    print(
        metadata[
            "Target_Forecast15"
        ].value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # SOURCE ANALYSIS
    # --------------------------------------------------------

    for source, group in metadata.groupby(
        "Source_File",
        sort=False
    ):

        group = group.sort_values(
            "Forecast_Start"
        ).reset_index(drop=True)

        n = len(group)

        train_end = int(
            n * TRAIN_RATIO
        )

        validation_end = int(
            n
            * (
                TRAIN_RATIO
                + VALIDATION_RATIO
            )
        )

        train_raw = group.iloc[
            :train_end
        ]

        validation_raw = group.iloc[
            train_end:validation_end
        ]

        test_raw = group.iloc[
            validation_end:
        ]

        # ----------------------------------------------------
        # APPLY EMBARGO
        # ----------------------------------------------------

        train_last = train_raw[
            "Forecast_Start"
        ].iloc[-1]

        validation_first_allowed = (
            train_last
            + pd.Timedelta(
                minutes=EMBARGO_MINUTES
            )
        )

        validation = validation_raw[
            validation_raw[
                "Forecast_Start"
            ]
            >= validation_first_allowed
        ]

        if len(validation) == 0:
            print(
                f"\nWARNING: no validation samples "
                f"remain for {source}"
            )
            continue

        validation_last = validation[
            "Forecast_Start"
        ].iloc[-1]

        test_first_allowed = (
            validation_last
            + pd.Timedelta(
                minutes=EMBARGO_MINUTES
            )
        )

        test = test_raw[
            test_raw[
                "Forecast_Start"
            ]
            >= test_first_allowed
        ]

        # ----------------------------------------------------
        # SOURCE HEADER
        # ----------------------------------------------------

        print("\n" + "-" * 80)
        print(f"SOURCE: {source}")
        print("-" * 80)

        print(
            f"Original sequences: {n}"
        )

        # ----------------------------------------------------
        # FUNCTION TO PRINT SPLIT
        # ----------------------------------------------------

        def describe_split(
            name,
            data
        ):

            print(
                f"\n{name}:"
            )

            print(
                f"  Samples: {len(data)}"
            )

            if len(data) == 0:
                return

            benign = int(
                np.sum(
                    data[
                        "Target_Forecast15"
                    ] == 0
                )
            )

            attack = int(
                np.sum(
                    data[
                        "Target_Forecast15"
                    ] == 1
                )
            )

            print(
                f"  Benign:  {benign}"
            )

            print(
                f"  Attack:  {attack}"
            )

            if len(data) > 0:

                print(
                    f"  Attack rate: "
                    f"{attack / len(data) * 100:.2f}%"
                )

                print(
                    f"  Time range: "
                    f"{data['Forecast_Start'].iloc[0]}"
                    f" → "
                    f"{data['Forecast_Start'].iloc[-1]}"
                )

            if (
                "Future_Attack_Types"
                in data.columns
            ):

                attack_rows = data[
                    data[
                        "Target_Forecast15"
                    ] == 1
                ]

                if len(attack_rows) > 0:

                    print(
                        "  Future attack types:"
                    )

                    counts = {}

                    for value in attack_rows[
                        "Future_Attack_Types"
                    ].astype(str):

                        for attack_type in value.split(
                            "|"
                        ):

                            if attack_type == "Benign":
                                continue

                            counts[
                                attack_type
                            ] = (
                                counts.get(
                                    attack_type,
                                    0
                                )
                                + 1
                            )

                    for attack_type, count in sorted(
                        counts.items(),
                        key=lambda item: item[1],
                        reverse=True
                    ):

                        print(
                            f"    {attack_type}: "
                            f"{count}"
                        )

        # ----------------------------------------------------
        # DESCRIBE
        # ----------------------------------------------------

        describe_split(
            "TRAIN",
            train_raw
        )

        describe_split(
            "VALIDATION",
            validation
        )

        describe_split(
            "TEST",
            test
        )

        # ----------------------------------------------------
        # GAPS
        # ----------------------------------------------------

        print(
            "\nTemporal gaps:"
        )

        train_last = train_raw[
            "Forecast_Start"
        ].iloc[-1]

        validation_first = validation[
            "Forecast_Start"
        ].iloc[0]

        validation_last = validation[
            "Forecast_Start"
        ].iloc[-1]

        test_first = test[
            "Forecast_Start"
        ].iloc[0]

        train_val_gap = (
            validation_first
            - train_last
        )

        val_test_gap = (
            test_first
            - validation_last
        )

        print(
            f"  Train → Validation: "
            f"{train_val_gap}"
        )

        print(
            f"  Validation → Test: "
            f"{val_test_gap}"
        )

        # ----------------------------------------------------
        # ATTACK TRANSITIONS
        # ----------------------------------------------------

        print(
            "\nAttack transitions:"
        )

        attack_series = group[
            "Target_Forecast15"
        ].to_numpy()

        transition_indices = []

        for i in range(
            1,
            len(attack_series)
        ):

            if (
                attack_series[i - 1] == 0
                and attack_series[i] == 1
            ):

                transition_indices.append(i)

        if transition_indices:

            for index in transition_indices:

                transition_time = group[
                    "Forecast_Start"
                ].iloc[index]

                print(
                    f"  Attack begins around: "
                    f"{transition_time}"
                )

        else:

            print(
                "  No 0 → 1 transitions found."
            )

    # --------------------------------------------------------
    # GLOBAL ATTACK TRANSITIONS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("GLOBAL ATTACK TRANSITION SUMMARY")
    print("=" * 80)

    all_transitions = []

    for source, group in metadata.groupby(
        "Source_File",
        sort=False
    ):

        group = group.sort_values(
            "Forecast_Start"
        ).reset_index(drop=True)

        values = group[
            "Target_Forecast15"
        ].to_numpy()

        for i in range(
            1,
            len(values)
        ):

            if (
                values[i - 1] == 0
                and values[i] == 1
            ):

                all_transitions.append(
                    {
                        "Source_File": source,
                        "Transition_Time": group[
                            "Forecast_Start"
                        ].iloc[i],
                    }
                )

    transitions_df = pd.DataFrame(
        all_transitions
    )

    if len(transitions_df) > 0:

        print(
            transitions_df.to_string(
                index=False
            )
        )

    else:

        print(
            "No attack transitions found."
        )

    # --------------------------------------------------------
    # FINAL DIAGNOSIS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("DIAGNOSTIC CONCLUSION")
    print("=" * 80)

    print(
        """
The purpose of this diagnostic is to determine whether the
chronological split produces meaningful validation and test
sets.

Pay particular attention to:

1. Validation benign/attack balance.
2. Attack periods crossing split boundaries.
3. Whether validation contains enough benign samples.
4. Whether the validation set contains multiple attack types.
5. The size of the temporal gaps between splits.
6. Whether the current split is representative of the
   underlying capture.

DO NOT retrain the model based on these results yet.
"""

    )

    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()