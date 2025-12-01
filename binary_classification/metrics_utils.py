import math
import logging

logging.basicConfig(level=logging.INFO)

def calculate_metrics(TP, FP, TN, FN):
    # Accuracy
    accuracy = (TP + TN) / (TP + FP + TN + FN)

    # Precision
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0

    # Recall
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    # F1 Score
    if (precision + recall) == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    }

def calculate_mcc(TP, FP, TN, FN):
    numerator = (TP * TN) - (FP * FN)
    denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    if denominator == 0:
        return 0  # MCC is undefined; return 0 or handle as needed
    else:
        return numerator / denominator

def print_metrics(TP, FP, TN, FN):
    """
    Utility function to calculate and print all metrics including MCC.
    """
    
    metrics = calculate_metrics(TP, FP, TN, FN)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    
    mcc = calculate_mcc(TP, FP, TN, FN)
    print(f"MCC: {mcc:.4f}")
