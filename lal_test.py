from sweep_base import LAL_Sweep

if __name__ == "__main__":
    cfg = {
        "learning_rate": 0.01,
        "num_leaves": 200,
        "min_data_in_leaf": 500,
        "bagging_fraction": 1,
        "feature_fraction": 1,
        "min_child_samples": 10,
    }
    
    lal_model = LAL_Sweep(cfg)
    lal_model.init()
    lal_model.run(1000)