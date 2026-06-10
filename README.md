# Titanic Survival Prediction Using Machine Learning Classifiers

## Project Information

| Field | Details |
| --- | --- |
| Intern ID | CITS4384 |
| Full Name | Aatish Ayyapath |
| No. of Days | 1 |
| Project Name | `Titanic Survival Prediction` |
| Project Scope | Build a machine learning pipeline that cleans Titanic passenger data, performs exploratory analysis, engineers meaningful features, trains a Random Forest Classifier, evaluates model performance, and generates prediction outputs for unseen test data. |

## Project Objective

This project predicts whether a passenger survived the Titanic disaster by using the Titanic dataset provided in the `data/` folder. The workflow compares a tuned Random Forest Classifier with a CatBoost Classifier and keeps the better-performing model for the final project outputs.

## Technical / Coding Checklist

- Complete project code is organized properly in separate files.
- Code includes comments and readable function-based structure.
- Files and folders are arranged neatly for GitHub submission.
- Repository is ready to upload to GitHub.
- A GitHub repository link can be submitted after you push the project.
- This project can be used as a unique repository for your technical/coding submission.

## Folder Structure

```text
Titanic_Survival_Prediction/
├── data/
│   ├── gender_submission.csv
│   ├── test.csv
│   └── train.csv
├── docs/
│   ├── documentation/
│   │   └── project_documentation.md
│   └── screenshots/
│       └── README.md
├── models/
├── outputs/
│   ├── data/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── src/
│   ├── config.py
│   ├── data_cleaning_and_exploration.py
│   ├── evaluation.py
│   ├── feature_extraction.py
│   └── train_test_model.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Files Overview

- `src/data_cleaning_and_exploration.py`: Loads raw data, performs basic cleaning, creates EDA summary files, and optionally saves charts.
- `src/feature_extraction.py`: Creates useful model features such as title, family size, deck, fare per person, and encoded categorical features.
- `src/train_test_model.py`: Splits the training data, tunes Random Forest and CatBoost, selects the better model, saves it, and produces predictions.
- `src/evaluation.py`: Computes accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and optional evaluation plots.
- `src/config.py`: Stores common file paths used across the project.

## Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn

## How to Run the Project

1. Create and activate a Python virtual environment.
2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Run the files in this order:

```bash
python src/data_cleaning_and_exploration.py
python src/feature_extraction.py
python src/train_test_model.py
python src/evaluation.py
```

## Output Files Generated

- Cleaned datasets in `outputs/data/`
- Feature-engineered datasets in `outputs/data/`
- Selected trained model file in `models/`
- Validation and test predictions in `outputs/predictions/`
- Evaluation metrics and reports in `outputs/metrics/`
- Charts and visual outputs in `outputs/figures/`

## Include in README File

This repository includes or is prepared to include:

- Source code
- README file
- Screenshots
- Output images
- Documentation

## Screenshots

Add execution screenshots in the `docs/screenshots/` folder after running the project. Suggested screenshots:

- Terminal output of each Python script
- Generated charts from `outputs/figures/`
- Final prediction files created in `outputs/predictions/`

## Documentation

Detailed project notes are available in:

- `docs/documentation/project_documentation.md`

## GitHub Upload Steps

1. Create a new GitHub repository with a unique project name.
2. Open a terminal in this project folder.
3. Run:

```bash
git init
git add .
git commit -m "Add Titanic Random Forest classifier project"
git branch -M main
git remote add origin <your-github-repository-url>
git push -u origin main
```

4. Copy the GitHub repository link and submit it as required.

## Notes

- The project currently selects the final model using cross-validation comparison between Random Forest and CatBoost.
- If `matplotlib` or `seaborn` are not installed, the scripts will still run, but plot images will be skipped.
- Replace the placeholder values in the Project Information table before submission.
