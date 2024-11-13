from sweep_base import Uncertainty_Sweep

if __name__ == "__main__":
    cfg = {
        "learning_rate": 0.01,
        "num_leaves": 200,
        "min_data_in_leaf": 500,
        "bagging_fraction": 1,
        "feature_fraction": 1,
        "min_child_samples": 10,
    }
    
    unc_model = Uncertainty_Sweep(cfg)
    unc_model.run(10000)
    # unc_model.eval()

