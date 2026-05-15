import pandas as pd
import numpy as np

# =====================================================
# LOAD DATA
# =====================================================

participants = pd.read_excel("dataset.xlsx", sheet_name="Participants")
raw = pd.read_excel("dataset.xlsx", sheet_name="Raw Data")

# =====================================================
# CLEANING
# =====================================================

raw = raw.sort_values(["essay_num", "op_index"])

raw["add"] = raw["add"].fillna("")
raw["delete"] = raw["delete"].fillna("")
raw["selected_text"] = raw["selected_text"].fillna("")

# =====================================================
# FEATURE EXTRACTION
# =====================================================

features = []

for essay_id, df in raw.groupby("essay_num"):

    df = df.sort_values("time").reset_index(drop=True)

    feat = {}
    feat["essay_id"] = essay_id

    # =================================================
    # BASIC COUNTS
    # =================================================

    feat["num_operations"] = len(df)

    insertions = df[df["op_type"] == "i"]
    deletions = df[df["op_type"] == "d"]

    feat["num_insertions"] = len(insertions)
    feat["num_deletions"] = len(deletions)

    # =================================================
    # CHARACTER CONTRIBUTIONS
    # =================================================

    feat["chars_inserted"] = (
        insertions["add"]
        .astype(str)
        .str.len()
        .sum()
    )

    feat["chars_deleted"] = (
        deletions["delete"]
        .astype(str)
        .str.len()
        .sum()
    )

    # =================================================
    # TIMING FEATURES
    # =================================================

    times = df["time"].values

    if len(times) > 1:
        deltas = np.diff(times)
    else:
        deltas = np.array([0])

    feat["mean_interaction_gap"] = np.mean(deltas)
    feat["median_interaction_gap"] = np.median(deltas)
    feat["max_pause"] = np.max(deltas)

    # Long pauses = possible ideation
    feat["num_long_pauses"] = np.sum(deltas > 10)

    # =================================================
    # WRITING BURSTS
    # =================================================

    burst_threshold = 5

    burst_lengths = []
    current_burst = 1

    for d in deltas:

        if d < burst_threshold:
            current_burst += 1
        else:
            burst_lengths.append(current_burst)
            current_burst = 1

    burst_lengths.append(current_burst)

    feat["mean_burst_length"] = np.mean(burst_lengths)
    feat["max_burst_length"] = np.max(burst_lengths)

    # =================================================
    # REVIEWING / CURSOR MOVEMENT
    # =================================================

    backward_cursor_moves = 0
    review_operations = 0

    prev_cursor = None

    for _, row in df.iterrows():

        cursor = str(row["cursor_location"])

        try:
            current_pos = int(
                cursor.split("[")[1].split(",")[0]
            )
        except:
            current_pos = None

        if prev_cursor is not None and current_pos is not None:

            # Moving backward in text
            if current_pos < prev_cursor:
                backward_cursor_moves += 1
                review_operations += 1

        prev_cursor = current_pos

    feat["backward_cursor_moves"] = backward_cursor_moves
    feat["review_operations"] = review_operations

    # =================================================
    # SMALL EDITS VS LARGE INSERTIONS
    # =================================================

    small_edits = 0
    large_insertions = 0

    for _, row in insertions.iterrows():

        text = str(row["add"])
        length = len(text)

        if length <= 3:
            small_edits += 1

        if length >= 20:
            large_insertions += 1

    feat["small_edits"] = small_edits
    feat["large_insertions"] = large_insertions

    # =================================================
    # AI-LIKE INSERTIONS
    # =================================================

    ai_like_insertions = 0
    ai_chars = 0

    for _, row in insertions.iterrows():

        text = str(row["add"])

        # Heuristic:
        # very large insertion = possible AI paste/acceptance
        if len(text) >= 30:

            ai_like_insertions += 1
            ai_chars += len(text)

    feat["ai_like_insertions"] = ai_like_insertions
    feat["ai_chars"] = ai_chars

    # =================================================
    # HUMAN VS AI CONTRIBUTION
    # =================================================

    total_chars = feat["chars_inserted"]

    human_chars = total_chars - ai_chars

    feat["human_chars"] = human_chars

    if total_chars > 0:

        feat["ai_ratio"] = ai_chars / total_chars
        feat["human_ratio"] = human_chars / total_chars

    else:

        feat["ai_ratio"] = 0
        feat["human_ratio"] = 0

    # =================================================
    # IDEATION VS DRAFTING TIME
    # =================================================

    ideation_time = deltas[deltas > 10].sum()
    drafting_time = deltas[deltas <= 10].sum()

    feat["ideation_time"] = ideation_time
    feat["drafting_time"] = drafting_time

    total_time = ideation_time + drafting_time

    if total_time > 0:

        feat["ideation_ratio"] = (
            ideation_time / total_time
        )

        feat["drafting_ratio"] = (
            drafting_time / total_time
        )

    else:

        feat["ideation_ratio"] = 0
        feat["drafting_ratio"] = 0

    # =================================================
    # STORE FEATURES
    # =================================================

    features.append(feat)

# =====================================================
# CREATE FEATURE TABLE
# =====================================================

feature_df = pd.DataFrame(features)

# =====================================================
# MERGE PARTICIPANT DATA
# =====================================================

feature_df = feature_df.merge(
    participants,
    left_on="essay_id",
    right_on="id",
    how="left"
)

# =====================================================
# SAVE
# =====================================================

feature_df.to_csv(
    "features.csv",
    index=False
)

print(feature_df.head())
