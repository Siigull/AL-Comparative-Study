import numpy as np
import wandb
from sklearn.metrics import classification_report


def wandb_report(training_f1, true_labels, pred_labels, target_names):
    report_columns = ["Class", "Precision", "Recall", "F1-score", "Support"]
    
    class_report = classification_report(true_labels, pred_labels, zero_division=0).splitlines()

    report_table = []
    for line in class_report[2:(len(target_names) + 2)]:
        report_table.append(line.split())

    table = wandb.Table(data=training_f1, columns=["round", "f1-score"])

    wandb.log({
        f"eval/report": wandb.Table(data=report_table, columns=report_columns),
        f"eval/training": wandb.plot.line(table, "round", "f1-score", ), 
    })