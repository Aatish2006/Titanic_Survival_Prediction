import json
import re

import numpy as np
import pandas as pd

from config import (
    CLEANED_TEST_FILE,
    CLEANED_TRAIN_FILE,
    FEATURE_SUMMARY_FILE,
    TEST_FEATURES_FILE,
    TRAIN_FEATURES_FILE,
    ensure_directories,
)


TITLE_PATTERN = re.compile(r",\s*([^\.]+)\.")


def extract_title(name: str) -> str:
    match = TITLE_PATTERN.search(name)
    if not match:
        return "Unknown"

    title = match.group(1).strip()
    title_map = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Royalty",
        "Countess": "Royalty",
        "Sir": "Royalty",
        "Don": "Royalty",
        "Dona": "Royalty",
        "Jonkheer": "Royalty",
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Dr": "Officer",
        "Rev": "Officer",
    }
    return title_map.get(title, title)


def extract_ticket_prefix(ticket: str) -> str:
    ticket = str(ticket).replace(".", "").replace("/", "").strip()
    parts = ticket.split()
    return parts[0] if len(parts) > 1 else "NONE"


def extract_deck(cabin: str) -> str:
    cabin = str(cabin).strip()
    if cabin == "Unknown" or cabin == "":
        return "U"
    return cabin[0]


def build_age_reference(train_df: pd.DataFrame) -> pd.Series:
    age_reference = (
        train_df.groupby(["Sex", "Pclass", "Title"], observed=False)["Age"]
        .median()
        .dropna()
    )
    return age_reference


def impute_age(row: pd.Series, age_reference: pd.Series, global_age_median: float) -> float:
    if pd.notna(row["Age"]):
        return float(row["Age"])

    lookup_key = (row["Sex"], row["Pclass"], row["Title"])
    if lookup_key in age_reference.index:
        return float(age_reference.loc[lookup_key])

    return float(global_age_median)


def fare_band(fare: float) -> str:
    if fare < 8:
        return "Low"
    if fare < 15:
        return "LowerMiddle"
    if fare < 32:
        return "UpperMiddle"
    return "High"


def age_band(age: float) -> str:
    if age < 16:
        return "Child"
    if age < 32:
        return "YoungAdult"
    if age < 48:
        return "Adult"
    if age < 64:
        return "Mature"
    return "Senior"


def family_group_label(family_size: int) -> str:
    if family_size == 1:
        return "Solo"
    if family_size <= 4:
        return "Small"
    return "Large"


def engineer_features(
    df: pd.DataFrame,
    ticket_counts: pd.Series,
    age_reference: pd.Series,
    global_age_median: float,
) -> pd.DataFrame:
    featured_df = df.copy()
    featured_df["Title"] = featured_df["Name"].apply(extract_title)
    featured_df["FamilySize"] = featured_df["SibSp"] + featured_df["Parch"] + 1
    featured_df["IsAlone"] = (featured_df["FamilySize"] == 1).astype(int)
    featured_df["AgeMissing"] = featured_df["Age"].isna().astype(int)
    featured_df["FareMissing"] = featured_df["Fare"].isna().astype(int)
    featured_df["Deck"] = featured_df["Cabin"].apply(extract_deck)
    featured_df["TicketPrefix"] = featured_df["Ticket"].apply(extract_ticket_prefix)
    featured_df["Age"] = featured_df.apply(
        impute_age,
        axis=1,
        age_reference=age_reference,
        global_age_median=global_age_median,
    )
    featured_df["Fare"] = featured_df.groupby("Pclass", observed=False)["Fare"].transform(
        lambda series: series.fillna(series.median())
    )
    featured_df["Fare"] = featured_df["Fare"].fillna(featured_df["Fare"].median())
    featured_df["FarePerPerson"] = featured_df["Fare"] / featured_df["FamilySize"].clip(lower=1)
    featured_df["NameLength"] = featured_df["Name"].str.len()
    featured_df["TicketGroupSize"] = featured_df["Ticket"].map(ticket_counts).fillna(1).astype(int)
    featured_df["CabinKnown"] = (featured_df["Deck"] != "U").astype(int)
    featured_df["FamilyGroup"] = featured_df["FamilySize"].apply(family_group_label)
    featured_df["AgeBand"] = featured_df["Age"].apply(age_band)
    featured_df["FareBand"] = featured_df["Fare"].apply(fare_band)
    featured_df["Pclass"] = featured_df["Pclass"].astype(str)
    featured_df["TicketGroupSize"] = np.clip(featured_df["TicketGroupSize"], 1, 8)

    columns_to_drop = ["Name", "Ticket", "Cabin"]
    featured_df = featured_df.drop(columns=columns_to_drop)
    return featured_df


def align_train_test_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_target = train_df["Survived"]
    train_ids = train_df["PassengerId"]
    test_ids = test_df["PassengerId"]

    train_features = train_df.drop(columns=["Survived"])
    combined = pd.concat([train_features, test_df], axis=0, ignore_index=True)
    combined = pd.get_dummies(
        combined,
        columns=[
            "Sex",
            "Embarked",
            "Title",
            "Deck",
            "TicketPrefix",
            "Pclass",
            "FamilyGroup",
            "AgeBand",
            "FareBand",
        ],
        drop_first=True,
        dtype=int,
    )

    encoded_train = combined.iloc[: len(train_df)].copy()
    encoded_test = combined.iloc[len(train_df) :].copy()

    encoded_train.insert(1, "Survived", train_target.values)
    encoded_train["PassengerId"] = train_ids.values
    encoded_test["PassengerId"] = test_ids.values

    return encoded_train, encoded_test


def main() -> None:
    ensure_directories()
    train_df = pd.read_csv(CLEANED_TRAIN_FILE)
    test_df = pd.read_csv(CLEANED_TEST_FILE)

    base_train = train_df.copy()
    base_test = test_df.copy()
    base_train["Title"] = base_train["Name"].apply(extract_title)
    base_test["Title"] = base_test["Name"].apply(extract_title)

    combined_tickets = pd.concat([base_train["Ticket"], base_test["Ticket"]], ignore_index=True)
    ticket_counts = combined_tickets.value_counts()
    age_reference = build_age_reference(base_train)
    global_age_median = float(base_train["Age"].median())

    engineered_train = engineer_features(base_train, ticket_counts, age_reference, global_age_median)
    engineered_test = engineer_features(base_test, ticket_counts, age_reference, global_age_median)
    aligned_train, aligned_test = align_train_test_features(engineered_train, engineered_test)

    aligned_train.to_csv(TRAIN_FEATURES_FILE, index=False)
    aligned_test.to_csv(TEST_FEATURES_FILE, index=False)

    feature_summary = {
        "train_feature_shape": list(aligned_train.shape),
        "test_feature_shape": list(aligned_test.shape),
        "feature_columns": aligned_train.columns.tolist(),
    }
    with open(FEATURE_SUMMARY_FILE, "w", encoding="utf-8") as file:
        json.dump(feature_summary, file, indent=4)

    print(f"Saved training features to: {TRAIN_FEATURES_FILE}")
    print(f"Saved test features to: {TEST_FEATURES_FILE}")


if __name__ == "__main__":
    main()
