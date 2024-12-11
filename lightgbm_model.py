import lightgbm as lgbt
import numpy as np
# import hydra
import wandb
from wandb_reporter import wandb_report

import os

from cesnet_datazoo.datasets import CESNET_TLS_Year22
from cesnet_datazoo.config import DatasetConfig, AppSelection, ValidationApproach
from sklearn.metrics import f1_score, accuracy_score

def f1_eval(preds, train_data):
    labels = train_data.get_label()
    
    preds = np.argmax(preds, axis=1)
    
    f1 = f1_score(labels, preds, average='macro')

    return 'f1', f1, True

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

    # cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(cfg["learning_rate"])

    try:
        dataset = CESNET_TLS_Year22(data_root="/storage/brno2/home/sigull/datasets/CESNET-TLS-Year22/", size="XS")
    except:
        dataset = CESNET_TLS_Year22(data_root="~/datasets/CESNET-TLS-Year22/", size="XS")
    nclasses = 180

    common_params = {
        "dataset" : dataset,
        "apps_selection" : AppSelection.ALL_KNOWN,
    }

    dataset_config = DatasetConfig(**common_params)
    dataset.set_dataset_config_and_initialize(dataset_config)
    train_dataframe = dataset.get_train_df(flatten_ppi=True)
    val_dataframe = dataset.get_val_df(flatten_ppi=True)
    test_dataframe = dataset.get_test_df(flatten_ppi=True)

    X = train_dataframe.drop(columns="APP").to_numpy()[:100000]
    y = train_dataframe["APP"].to_numpy()[:100000]

    X_val = val_dataframe.drop(columns="APP").to_numpy()[:100000]
    y_val = val_dataframe["APP"].to_numpy()[:100000]
    
    X_test = test_dataframe.drop(columns="APP").to_numpy()[:100000]
    y_test = test_dataframe["APP"].to_numpy()[:100000]

    train_data = lgbt.Dataset(X, y)
    val_data = lgbt.Dataset(X_val, y_val, reference=train_data)

    nefunguje pip install verze
    ručně build 
    jinej build vzdáleně než lokálně 
    killed 
    add module

    lgbm_params = {
        "verbose": -1,
        "num_threads": -1,
        "max_bin": 63,
        "device": "gpu",
        "gpu_device_id": 0,
        "objective": "multiclass",
        "num_class": nclasses,
        "metric": "multi_logloss",
        "boosting_type": "dart",
        "learning_rate": cfg["learning_rate"],
        "num_leaves": cfg["num_leaves"],
        "min_data_in_leaf": cfg["min_data_in_leaf"],
        "bagging_fraction": cfg["bagging_fraction"],
        "feature_fraction": cfg["feature_fraction"],
        "min_child_samples": cfg["min_child_samples"],
        "max_depth": -1,
        "seed": 10
    }

    evals_result = {}
    model = lgbt.train(
        params=lgbm_params,
        train_set=train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        num_boost_round=30,  
        callbacks=[
            lgbt.early_stopping(stopping_rounds=50),
            lgbt.log_evaluation(10),
            lgbt.record_evaluation(evals_result),
        ],
        feval=f1_eval,
    )

    predict_arr = model.predict(X_test)
    predict_arr = np.argmax(predict_arr, axis=1)

    # print(evals_result)

    nclasses = len(np.unique(y_test))

    eval_arr = [[index, value] for index, value in enumerate(evals_result['val']['f1'])]

    f1 = f1_score(y_test, predict_arr, average='macro')

    wandb.log({"eval/f1": f1})

    wandb_report(eval_arr, y_test, predict_arr, [str(i) for i in range(nclasses)])
    
    print("done")

if __name__ == "__main__":
    main()
