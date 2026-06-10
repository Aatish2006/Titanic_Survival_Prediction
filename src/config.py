from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DATA_DIR = OUTPUTS_DIR / "data"
OUTPUTS_METRICS_DIR = OUTPUTS_DIR / "metrics"
OUTPUTS_PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
OUTPUTS_FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
SCREENSHOTS_DIR = DOCS_DIR / "screenshots"
DOCUMENTATION_DIR = DOCS_DIR / "documentation"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "gender_submission.csv"

CLEANED_TRAIN_FILE = OUTPUTS_DATA_DIR / "train_cleaned.csv"
CLEANED_TEST_FILE = OUTPUTS_DATA_DIR / "test_cleaned.csv"
TRAIN_FEATURES_FILE = OUTPUTS_DATA_DIR / "train_features.csv"
TEST_FEATURES_FILE = OUTPUTS_DATA_DIR / "test_features.csv"
EDA_SUMMARY_FILE = OUTPUTS_DATA_DIR / "eda_summary.json"
FEATURE_SUMMARY_FILE = OUTPUTS_DATA_DIR / "feature_summary.json"

MODEL_FILE = MODELS_DIR / "titanic_classifier.joblib"
TRAINING_SUMMARY_FILE = OUTPUTS_METRICS_DIR / "training_summary.json"
CLASSIFICATION_REPORT_FILE = OUTPUTS_METRICS_DIR / "classification_report.txt"
EVALUATION_METRICS_FILE = OUTPUTS_METRICS_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_FILE = OUTPUTS_METRICS_DIR / "confusion_matrix.csv"
FEATURE_IMPORTANCE_FILE = OUTPUTS_METRICS_DIR / "feature_importance.csv"
VALIDATION_PREDICTIONS_FILE = OUTPUTS_PREDICTIONS_DIR / "validation_predictions.csv"
TEST_PREDICTIONS_FILE = OUTPUTS_PREDICTIONS_DIR / "test_predictions.csv"
SUBMISSION_FILE = OUTPUTS_PREDICTIONS_DIR / "submission.csv"


def ensure_directories() -> None:
    directories = [
        OUTPUTS_DIR,
        OUTPUTS_DATA_DIR,
        OUTPUTS_METRICS_DIR,
        OUTPUTS_PREDICTIONS_DIR,
        OUTPUTS_FIGURES_DIR,
        MODELS_DIR,
        DOCS_DIR,
        SCREENSHOTS_DIR,
        DOCUMENTATION_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
