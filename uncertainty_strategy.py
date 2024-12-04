from sweep_base import Uncertainty_Sweep
import wandb
from wandb_reporter import wandb_report

def main() -> None:
    # wandb.init()
    cfg = {
        "learning_rate": 0.01,
        "num_leaves": 200,
        "min_data_in_leaf": 500,
        "bagging_fraction": 1,
        "feature_fraction": 1,
        "min_child_samples": 10,
        "average_unc_cutoff": 0.06
    }
    # cfg = wandb.config

    unc_model = Uncertainty_Sweep(cfg)
    unc_model.run(10000)
    f1, eval_arr, y_test, predict_arr, classes = unc_model.eval()

    wandb.log({"eval/f1": f1})

    wandb_report(eval_arr, y_test, predict_arr, classes)

if __name__ == "__main__":
    main()