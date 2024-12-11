from sweep_base import Random_Sweep

if __name__ == "__main__":
    cfg = {
        "learning_rate": 0.01,
        "num_leaves": 200,
        "min_data_in_leaf": 500,
        "bagging_fraction": 1,
        "feature_fraction": 1,
        "min_child_samples": 10,
        "random_frac": 0.05,
    }
    
    unc_model = Random_Sweep(cfg)
    unc_model.run()
    # unc_model.eval()

