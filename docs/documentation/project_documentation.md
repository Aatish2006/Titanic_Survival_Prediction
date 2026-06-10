# Project Documentation

## Project Title

Titanic Survival Prediction Using Machine Learning Classifiers

## Problem Statement

The goal of this project is to predict passenger survival on the Titanic by using supervised machine learning on the provided training data.

## Workflow Summary

1. Raw data is read from the `data/` folder.
2. Basic cleaning is applied in the data cleaning script.
3. Exploratory summary information is saved for review.
4. Feature engineering creates model-friendly columns.
5. Random Forest and CatBoost models are tuned and compared on the prepared dataset.
6. The better model is selected and evaluated on a validation split.
7. Prediction and report files are saved for future use and GitHub presentation.

## Feature Engineering Used

- Passenger title extracted from name
- Family size
- Is alone flag
- Age missing flag
- Fare missing flag
- Deck extracted from cabin
- Ticket prefix extraction
- Fare per person
- Name length
- One-hot encoding for categorical variables

## Models Compared

- Random Forest Classifier
- CatBoost Classifier
- Final selection is based on model comparison after hyperparameter tuning and cross-validation.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## Deliverables

- Organized source code
- Model artifact
- Predictions
- Evaluation reports
- Figures
- README
- Documentation
- Screenshot folder for submission proof
