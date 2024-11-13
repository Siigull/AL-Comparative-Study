from cesnet_datazoo.datasets import CESNET_TLS_Year22
from cesnet_datazoo.config import DatasetConfig, AppSelection, ValidationApproach
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import wandb

from wandb_reporter import wandb_report

iters = 300

if __name__ == "__main__":
    dataset = CESNET_TLS_Year22(data_root="~/datasets/CESNET-TLS-Year22/", size="XS")

    common_params = {
        "dataset" : dataset,
        "apps_selection" : AppSelection.ALL_KNOWN,
    }

    wandb.init(project="pelanekd-ALcompstudy-test-1")

    dataset_config = DatasetConfig(**common_params)
    dataset.set_dataset_config_and_initialize(dataset_config)
    train_dataframe = dataset.get_train_df(flatten_ppi=True)
    val_dataframe = dataset.get_val_df(flatten_ppi=True)
    test_dataframe = dataset.get_test_df(flatten_ppi=True)

    X = train_dataframe.drop(columns="APP").to_numpy()
    y = train_dataframe["APP"].to_numpy()

    X_test = test_dataframe.drop(columns="APP").to_numpy()[:100000]
    y_test = test_dataframe["APP"].to_numpy()[:100000]

    nfeatures = X.shape[1]

    ch_i = 0
    X_chosen = np.ndarray(shape = (iters, nfeatures))
    y_chosen = np.ndarray(shape = (iters,))

    training_results = []
    nclasses = len(test_dataframe.groupby("APP"))

    for i in range(1, iters):
        random_index = np.random.randint(0, len(X))
        X_chosen[ch_i] = X[random_index]
        y_chosen[ch_i] = y[random_index]
        np.delete(X, random_index)
        np.delete(y, random_index)
        ch_i += 1

        if i % 100 == 0:
            clf = RandomForestClassifier()
            clf.fit(X_chosen[:ch_i], y_chosen[:ch_i])
            predict_array = clf.predict(X_test)
            training_results.append([i, f1_score(predict_array, y_test, average="macro")])

    clf = RandomForestClassifier()
    clf.fit(X_chosen[:ch_i], y_chosen[:ch_i])
    wandb_report(training_results, y_test, clf.predict(X_test), [str(i) for i in range(nclasses)])
