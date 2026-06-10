import json

import pandas as pd

from config import (
    CLEANED_TEST_FILE,
    CLEANED_TRAIN_FILE,
    EDA_SUMMARY_FILE,
    OUTPUTS_FIGURES_DIR,
    TEST_FILE,
    TRAIN_FILE,
    ensure_directories,
)


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    return train_df, test_df


def clean_dataframe(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    cleaned_df = df.copy()
    cleaned_df.columns = [column.strip() for column in cleaned_df.columns]
    cleaned_df["Name"] = cleaned_df["Name"].fillna("Unknown").str.strip()
    cleaned_df["Sex"] = cleaned_df["Sex"].fillna("unknown").str.lower().str.strip()
    cleaned_df["Ticket"] = cleaned_df["Ticket"].fillna("UNKNOWN").str.strip()
    cleaned_df["Cabin"] = cleaned_df["Cabin"].fillna("Unknown").str.strip()
    cleaned_df["Embarked"] = cleaned_df["Embarked"].fillna(cleaned_df["Embarked"].mode().iloc[0]).str.upper()

    if "Fare" in cleaned_df.columns:
        cleaned_df["Fare"] = cleaned_df["Fare"].fillna(cleaned_df["Fare"].median())

    if is_train:
        cleaned_df = cleaned_df.drop_duplicates(subset=["PassengerId"]).reset_index(drop=True)

    return cleaned_df


def build_eda_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    return {
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "train_columns": train_df.columns.tolist(),
        "test_columns": test_df.columns.tolist(),
        "train_missing_values": train_df.isna().sum().to_dict(),
        "test_missing_values": test_df.isna().sum().to_dict(),
        "survival_distribution": train_df["Survived"].value_counts().sort_index().to_dict(),
        "sex_distribution": train_df["Sex"].value_counts().to_dict(),
        "embarked_distribution": train_df["Embarked"].value_counts(dropna=False).to_dict(),
    }


def save_optional_plots(train_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed. Skipping EDA plots.")
        return

    survival_plot = OUTPUTS_FIGURES_DIR / "survival_distribution.png"
    missing_plot = OUTPUTS_FIGURES_DIR / "missing_values.png"

    plt.figure(figsize=(6, 4))
    sns.countplot(data=train_df, x="Survived", hue="Survived", palette="Set2", legend=False)
    plt.title("Survival Distribution")
    plt.xlabel("Survived")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(survival_plot, dpi=200)
    plt.close()

    missing_values = train_df.isna().sum().sort_values(ascending=False)
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        plt.figure(figsize=(8, 4))
        sns.barplot(x=missing_values.index, y=missing_values.values, hue=missing_values.index, palette="crest", legend=False)
        plt.title("Missing Values in Training Data")
        plt.xlabel("Column")
        plt.ylabel("Missing Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(missing_plot, dpi=200)
        plt.close()


def main() -> None:
    ensure_directories()
    train_df, test_df = load_datasets()
    cleaned_train_df = clean_dataframe(train_df, is_train=True)
    cleaned_test_df = clean_dataframe(test_df, is_train=False)

    cleaned_train_df.to_csv(CLEANED_TRAIN_FILE, index=False)
    cleaned_test_df.to_csv(CLEANED_TEST_FILE, index=False)

    eda_summary = build_eda_summary(cleaned_train_df, cleaned_test_df)
    with open(EDA_SUMMARY_FILE, "w", encoding="utf-8") as file:
        json.dump(eda_summary, file, indent=4)

    save_optional_plots(cleaned_train_df)
    print(f"Saved cleaned training data to: {CLEANED_TRAIN_FILE}")
    print(f"Saved cleaned test data to: {CLEANED_TEST_FILE}")
    print(f"Saved EDA summary to: {EDA_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
