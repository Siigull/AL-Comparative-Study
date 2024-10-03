import numpy as np
import wandb
from sklearn.metrics import classification_report


def wandb_report(training_metrics, true_labels, pred_labels, target_names):
    report_columns = ["Class", "Precision", "Recall", "F1-score", "Support"]
    
    class_report = classification_report(true_labels, pred_labels, target_names=target_names, zero_division=0).splitlines()

    report_table = []
    for line in class_report[2:(len(target_names) + 2)]:
        report_table.append(line.split())

    wandb.log({
        f"eval/cm": wandb.plot.confusion_matrix(y_true=true_labels, preds=pred_labels, class_names=target_names),
        f"eval/report": wandb.Table(data=report_table, columns=report_columns),
        f"eval/training": wandb.Table(data=training_metrics, columns=["iters", "f1-score"]), 
    })