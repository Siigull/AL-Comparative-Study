from sweep_base import Uncertainty_Sweep
import wandb

def main():
    cfg = {
        "learning_rate": 0.01,
        "num_leaves": 200,
        "min_data_in_leaf": 500,
        "bagging_fraction": 1,
        "feature_fraction": 1,
        "min_child_samples": 10,
        "average_unc_cutoff": 0.06
    }

    model = Uncertainty_Sweep(cfg)
    model.run(10000)
    model.eval()

if __name__ == "__main__":
    main()