import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import (
    CLASSIFICATION_REPORT_FILE,
    CONFUSION_MATRIX_FILE,
    EVALUATION_METRICS_FILE,
    FEATURE_IMPORTANCE_FILE,
    OUTPUTS_FIGURES_DIR,
    VALIDATION_PREDICTIONS_FILE,
    ensure_directories,
)


def save_optional_evaluation_plots(confusion_df: pd.DataFrame, feature_importance_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed. Skipping evaluation plots.")
        return

    confusion_plot = OUTPUTS_FIGURES_DIR / "confusion_matrix.png"
    importance_plot = OUTPUTS_FIGURES_DIR / "top_feature_importance.png"

    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(confusion_plot, dpi=200)
    plt.close()

    top_features = feature_importance_df.head(10).sort_values("Importance")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=top_features, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig(importance_plot, dpi=200)
    plt.close()


def main() -> None:
    ensure_directories()
    validation_df = pd.read_csv(VALIDATION_PREDICTIONS_FILE)
    feature_importance_df = pd.read_csv(FEATURE_IMPORTANCE_FILE)

    y_true = validation_df["Actual"]
    y_pred = validation_df["Predicted"]
    y_score = validation_df["SurvivalProbability"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
    }

    metrics["note"] = "These metrics are from the held-out validation split. Cross-validation metrics are saved in outputs/metrics/training_summary.json."

    confusion = confusion_matrix(y_true, y_pred)
    confusion_df = pd.DataFrame(
        confusion,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    )
    confusion_df.to_csv(CONFUSION_MATRIX_FILE)

    report = classification_report(y_true, y_pred)
    with open(CLASSIFICATION_REPORT_FILE, "w", encoding="utf-8") as file:
        file.write(report)

    with open(EVALUATION_METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    save_optional_evaluation_plots(confusion_df, feature_importance_df)
    print(f"Saved evaluation metrics to: {EVALUATION_METRICS_FILE}")
    print(f"Saved classification report to: {CLASSIFICATION_REPORT_FILE}")


if __name__ == "__main__":
    main()
