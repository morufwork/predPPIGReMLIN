import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # --- Load & clean dataset ---
    df = pd.read_csv("classical_residue_features.csv", low_memory=False)

    # Convert all feature columns to numeric (fix mixed types)
    for col in df.columns[:-1]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna().reset_index(drop=True)

    # Features/labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # --- Model ---
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # --- 10-fold Stratified CV ---
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro"
    }

    cv_results = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    # --- Print per-fold ---
    print("Per-fold metrics:")
    for i in range(10):
        print(
            f"Fold {i+1}: "
            f"acc={cv_results['test_accuracy'][i]:.4f}, "
            f"prec_macro={cv_results['test_precision_macro'][i]:.4f}, "
            f"recall_macro={cv_results['test_recall_macro'][i]:.4f}, "
            f"f1_macro={cv_results['test_f1_macro'][i]:.4f}"
        )

    print("\nMean ± SD across folds:")
    for k, v in scoring.items():
        scores = cv_results[f"test_{v}"]
        print(f"{k}: {scores.mean():.4f} ± {scores.std():.4f}")

    # --- Out-of-fold predictions ---
    y_oof = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

    acc_oof = accuracy_score(y, y_oof)

    # --- Rounded classification report (4 decimals) ---
    report_dict = classification_report(y, y_oof, output_dict=True)

    rounded_report = {
        label: (
            {metric: round(value, 4) for metric, value in scores.items()}
            if isinstance(scores, dict)
            else round(scores, 4)
        )
        for label, scores in report_dict.items()
    }

    report_df = pd.DataFrame(rounded_report).T

    print(f"\nOOF Accuracy (10-fold): {acc_oof:.4f}\n")
    print("OOF Classification Report (rounded to 4 decimals):")
    print(report_df.to_string(float_format='%.4f'))


if __name__ == "__main__":
    main()
