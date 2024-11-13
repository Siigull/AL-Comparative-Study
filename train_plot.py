import lightgbm as lgbt

from cesnet_datazoo.datasets import CESNET_TLS_Year22
from cesnet_datazoo.config import DatasetConfig, AppSelection, ValidationApproach
from sklearn.metrics import f1_score, accuracy_score

import numpy as np

def f1_eval(preds, train_data):
    labels = train_data.get_label()
    
    preds = np.argmax(preds, axis=1)
    
    f1 = f1_score(labels, preds, average='macro')

    return 'f1', f1, True

def main():
    dataset = CESNET_TLS_Year22(data_root="~/datasets/CESNET-TLS-Year22/", size="XS")

    keys_arr = list(dataset.time_periods.keys())
    day_arr = []
    for i in keys_arr[10:47]:
        day_arr += dataset.time_periods[i]
    
    print(day_arr)

    nclasses = 180

    common_params = {
        "dataset" : dataset,
        "apps_selection" : AppSelection.ALL_KNOWN,
        "train_period_name": "W-2022-11",
        "test_period_name": "M-2022-12",
    }

    dataset_config = DatasetConfig(**common_params)
    dataset.set_dataset_config_and_initialize(dataset_config)
    train_dataframe = dataset.get_train_df(flatten_ppi=True)
    val_dataframe = dataset.get_val_df(flatten_ppi=True)
    test_dataframe = dataset.get_test_df(flatten_ppi=True)

    X = train_dataframe.drop(columns="APP").to_numpy()
    y = train_dataframe["APP"].to_numpy()

    X_val = val_dataframe.drop(columns="APP").to_numpy()
    y_val = val_dataframe["APP"].to_numpy()

    X_test = test_dataframe.drop(columns="APP").to_numpy()
    y_test = test_dataframe["APP"].to_numpy()

    lgbm_params = {
        "verbose": -1,
        "num_threads": -1,
        "objective": "multiclass",
        "num_class": nclasses,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "num_leaves": 200,
        "min_data_in_leaf": 500,
        "bagging_fraction": 1,
        "feature_fraction": 1,
        "min_child_samples": 10,
        "max_depth": -1,
        "seed": 10
    }

    amounts = [100_000, 500_000, 1_000_000, len(X)]

    evals = []

    for iters in amounts:
        train_data = lgbt.Dataset(X[:iters], y[:iters])
        val_data = lgbt.Dataset(X_val[:int(iters/5)], y_val[:int(iters/5)], reference=train_data)
        evals_result = {}
        model = lgbt.train(
            params=lgbm_params,
            train_set=train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            num_boost_round=700,  
            callbacks=[
                lgbt.early_stopping(stopping_rounds=50),
                lgbt.log_evaluation(10),
                lgbt.record_evaluation(evals_result),
            ],
            feval=f1_eval,
        )

        predict_arr = model.predict(X_test)
        predict_arr = np.argmax(predict_arr, axis=1)

        f1 = f1_score(y_test, predict_arr, average='macro')

        evals.append(f1)

    print(evals)


if __name__ == "__main__":
    main()
    