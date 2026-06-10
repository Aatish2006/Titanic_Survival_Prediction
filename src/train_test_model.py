import json

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

from config import (
    FEATURE_IMPORTANCE_FILE,
    MODEL_FILE,
    SUBMISSION_FILE,
    TEST_FEATURES_FILE,
    TEST_PREDICTIONS_FILE,
    TRAIN_FEATURES_FILE,
    TRAINING_SUMMARY_FILE,
    VALIDATION_PREDICTIONS_FILE,
    ensure_directories,
)


def summarize_cross_validation(cv_scores: dict) -> dict:
    return {
        "mean_scores": {
            "accuracy": float(cv_scores["test_accuracy"].mean()),
            "precision": float(cv_scores["test_precision"].mean()),
            "recall": float(cv_scores["test_recall"].mean()),
            "f1": float(cv_scores["test_f1"].mean()),
            "roc_auc": float(cv_scores["test_roc_auc"].mean()),
        },
        "std_scores": {
            "accuracy": float(cv_scores["test_accuracy"].std()),
            "precision": float(cv_scores["test_precision"].std()),
            "recall": float(cv_scores["test_recall"].std()),
            "f1": float(cv_scores["test_f1"].std()),
            "roc_auc": float(cv_scores["test_roc_auc"].std()),
        },
    }


def build_random_forest_search(inner_cv: StratifiedKFold) -> GridSearchCV:
    base_model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced_subsample",
    )
    param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [6, 8, 12, None],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5, None],
    }
    return GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )


def build_catboost_search(inner_cv: StratifiedKFold) -> GridSearchCV:
    base_model = CatBoostClassifier(
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
        loss_function="Logloss",
    )
    param_grid = {
        "iterations": [200, 400],
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "l2_leaf_reg": [3, 5, 7],
    }
    return GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )


def choose_better_model(model_results: list[dict]) -> dict:
    return sorted(
        model_results,
        key=lambda result: (
            result["cross_validation"]["mean_scores"]["roc_auc"],
            result["cross_validation"]["mean_scores"]["accuracy"],
            result["holdout_validation"]["accuracy"],
        ),
        reverse=True,
    )[0]


def main() -> None:
    ensure_directories()
    train_df = pd.read_csv(TRAIN_FEATURES_FILE)
    test_df = pd.read_csv(TEST_FEATURES_FILE)

    x = train_df.drop(columns=["PassengerId", "Survived"])
    y = train_df["Survived"]
    test_ids = test_df["PassengerId"]
    x_test = test_df.drop(columns=["PassengerId"])

    x_train, x_valid, y_train, y_valid, train_ids, valid_ids = train_test_split(
        x,
        y,
        train_df["PassengerId"],
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_searches = {
        "RandomForestClassifier": build_random_forest_search(inner_cv),
        "CatBoostClassifier": build_catboost_search(inner_cv),
    }
    model_results = []

    for model_name, grid_search in model_searches.items():
        grid_search.fit(x_train, y_train)
        model = grid_search.best_estimator_

        validation_predictions = model.predict(x_valid)
        validation_probabilities = model.predict_proba(x_valid)[:, 1]
        cv_scores = cross_validate(
            model,
            x,
            y,
            cv=outer_cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            n_jobs=-1,
        )

        model_results.append(
            {
                "model_name": model_name,
                "estimator": model,
                "best_hyperparameters": grid_search.best_params_,
                "grid_search_best_score": float(grid_search.best_score_),
                "holdout_validation": {
                    "accuracy": float(accuracy_score(y_valid, validation_predictions)),
                    "roc_auc": float(roc_auc_score(y_valid, validation_probabilities)),
                },
                "cross_validation": summarize_cross_validation(cv_scores),
            }
        )

    selected_result = choose_better_model(model_results)
    model = selected_result["estimator"]

    validation_predictions = model.predict(x_valid)
    validation_probabilities = model.predict_proba(x_valid)[:, 1]
    test_predictions = model.predict(x_test)

    validation_output = pd.DataFrame(
        {
            "PassengerId": valid_ids.values,
            "Actual": y_valid.values,
            "Predicted": validation_predictions,
            "SurvivalProbability": validation_probabilities,
        }
    ).sort_values("PassengerId")
    validation_output.to_csv(VALIDATION_PREDICTIONS_FILE, index=False)

    test_output = pd.DataFrame(
        {
            "PassengerId": test_ids,
            "PredictedSurvived": test_predictions,
        }
    )
    test_output.to_csv(TEST_PREDICTIONS_FILE, index=False)

    submission = pd.DataFrame(
        {
            "PassengerId": test_ids,
            "Survived": test_predictions,
        }
    )
    submission.to_csv(SUBMISSION_FILE, index=False)

    feature_importance = pd.DataFrame(
        {
            "Feature": x.columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

    training_summary = {
        "selected_model_name": selected_result["model_name"],
        "random_state": 42,
        "train_rows": int(len(x_train)),
        "validation_rows": int(len(x_valid)),
        "feature_count": int(x.shape[1]),
        "selection_metric": "cross_validation_mean_roc_auc",
        "best_hyperparameters": selected_result["best_hyperparameters"],
        "grid_search_best_score": selected_result["grid_search_best_score"],
        "holdout_validation_accuracy": selected_result["holdout_validation"]["accuracy"],
        "holdout_validation_roc_auc": selected_result["holdout_validation"]["roc_auc"],
        "cross_validation_mean_scores": selected_result["cross_validation"]["mean_scores"],
        "cross_validation_std_scores": selected_result["cross_validation"]["std_scores"],
        "model_comparison": [
            {
                "model_name": result["model_name"],
                "best_hyperparameters": result["best_hyperparameters"],
                "grid_search_best_score": result["grid_search_best_score"],
                "holdout_validation": result["holdout_validation"],
                "cross_validation_mean_scores": result["cross_validation"]["mean_scores"],
                "cross_validation_std_scores": result["cross_validation"]["std_scores"],
            }
            for result in model_results
        ],
        "hyperparameters": model.get_params(),
    }
    with open(TRAINING_SUMMARY_FILE, "w", encoding="utf-8") as file:
        json.dump(training_summary, file, indent=4, default=str)

    joblib.dump(model, MODEL_FILE)
    print(f"Saved trained model to: {MODEL_FILE}")
    print(f"Saved validation predictions to: {VALIDATION_PREDICTIONS_FILE}")
    print(f"Saved Kaggle-style submission to: {SUBMISSION_FILE}")


if __name__ == "__main__":
    main()
