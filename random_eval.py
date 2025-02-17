from sweep_base import Random_Sweep
import wandb
from wandb_reporter import wandb_report

def main():
    wandb.init(project="pelanekd-ALcompstudy", name="random_eval_run")

    cfg = {
        "learning_rate": 0.1,
        "num_leaves": 40,
        "min_data_in_leaf": 2806,
        "bagging_fraction": 0.8351,
        "feature_fraction": 0.9896,
        "min_child_samples": 16,
        "random_frac": 0.1,
    }

    wandb.config = cfg

    ran_model = Random_Sweep(cfg, eval=True)
    ran_model.run()
    f1, eval_arr, y_test, predict_arr, classes = ran_model.eval()

    wandb.log({"eval/f1": f1})

    wandb_report(eval_arr, y_test, predict_arr, classes)

    wandb.finish()

if __name__ == "__main__":
    main()