import lightgbm as lgbt
import numpy as np
import pandas as pd
# import hydra
import wandb
from wandb_reporter import wandb_report

import os
import sys

from cesnet_datazoo.datasets import CESNET_TLS_Year22
from cesnet_datazoo.config import DatasetConfig, AppSelection, ValidationApproach
from sklearn.metrics import f1_score, accuracy_score

def f1_eval(preds, train_data):
    labels = train_data.get_label()
    
    preds = np.argmax(preds, axis=1)
    
    f1 = f1_score(labels, preds, average='macro')

    return 'f1', f1, True

class Sweep_Class:
    def run():
        pass

    def eval():
        pass

    def __init__(self, cfg):
        self.cfg = cfg

        try:
            dataset = CESNET_TLS_Year22(data_root="/storage/brno2/home/sigull/datasets/CESNET-TLS-Year22/", size="L")
        except:
            dataset = CESNET_TLS_Year22(data_root="~/datasets/CESNET-TLS-Year22/", size="XS")
        
        self.nclasses = 180

        self.week_arr = list(dataset.time_periods.keys())[10:30]
        self.day_arr = []
        for i in self.week_arr:
            self.day_arr += dataset.time_periods[i]

        self.period = "day"
        self.day_i = 0
        self.week_i = 0

        common_params = {
            "dataset" : dataset,
            "apps_selection" : AppSelection.ALL_KNOWN,
            "train_period_name": self.week_arr[0],
            "train_dates": {self.day_arr[0]},
            "test_period_name": "M-2022-12",
        }

        dataset_config = DatasetConfig(**common_params)
        dataset.set_dataset_config_and_initialize(dataset_config)
        train_dataframe = dataset.get_train_df(flatten_ppi=True)
        val_dataframe = dataset.get_val_df(flatten_ppi=True)
        test_dataframe = dataset.get_test_df(flatten_ppi=True)

        for i in range(11, 7, -1):
            common_params = {
                "dataset" : dataset,
                "apps_selection" : AppSelection.ALL_KNOWN,
                "train_period_name": self.week_arr[0],
                "train_dates": {self.day_arr[0]},
                "test_period_name": f"M-2022-{str(i)}",
            }
            dataset_config = DatasetConfig(**common_params)
            dataset.set_dataset_config_and_initialize(dataset_config)
            test_dataframe = pd.concat([test_dataframe, dataset.get_test_df(flatten_ppi=True)])

        self.global_labels = dataset.available_classes
        self.label_to_global_y = {app: i for i, app in enumerate(self.global_labels)}
        known_apps = dataset.get_known_apps()
        local_to_global = {i: self.label_to_global_y[app] for i, app in enumerate(known_apps)}

        self.X = train_dataframe.drop(columns="APP").to_numpy()
        self.y = train_dataframe["APP"].to_numpy()
        self.y = np.array([local_to_global[y] for y in self.y])

        self.X_val = val_dataframe.drop(columns="APP").to_numpy()
        self.y_val = val_dataframe["APP"].to_numpy()
        self.y_val = np.array([local_to_global[y] for y in self.y_val])

        self.X_test = test_dataframe.drop(columns="APP").to_numpy()
        self.y_test = test_dataframe["APP"].to_numpy()
        self.y_test = np.array([local_to_global[y] for y in self.y_test])

        self.week_evals = []

        already_chosen = 15000

        self.chosen_indices = np.array([i for i in range(already_chosen)])
        self.unknown_indices = np.array([i for i in range(already_chosen, len(self.X))])
        self.prev_Xs = []
        self.prev_ys = []
        self.prev_valXs = []
        self.prev_valys = []

    def get_current_train_dataset_len(self):
        dataset_len = len(self.chosen_indices)

        try:
            for array_index in range(len(self.prev_Xs)-1, max(len(self.prev_Xs)-10, -1), -1):
                dataset_len += len(self.prev_Xs[array_index])
            
        except Exception as exception:
            print(exception)

        return dataset_len

    def train(self, max_rounds=1000):
        X_arr = np.copy(self.X[self.chosen_indices])
        y_arr = np.copy(self.y[self.chosen_indices])
        X_val_arr = np.copy(self.X_val)
        y_val_arr = np.copy(self.y_val)

        if self.period == "day":
            try:
                for array_index in range(len(self.prev_Xs)-1, max(len(self.prev_Xs)-7, -1), -1):
                    X_arr = np.concatenate([X_arr, self.prev_Xs[array_index]])
                    y_arr = np.concatenate([y_arr, self.prev_ys[array_index]])
                    X_val_arr = np.concatenate([X_val_arr, self.prev_valXs[array_index]])
                    y_val_arr = np.concatenate([y_val_arr, self.prev_valys[array_index]])
                
            except Exception as exception:
                print(exception)

        train_data = lgbt.Dataset(X_arr, y_arr)
        val_data = lgbt.Dataset(X_val_arr, y_val_arr, reference=train_data)

        lgbm_params = {
            "verbose": -1,
            "num_threads": -1,
            "max_bin": 63,
            "device": "cpu",
            "objective": "multiclass",
            "num_class": self.nclasses,
            "metric": "multi_logloss",
            "max_depth": 20,
            "learning_rate": self.cfg["learning_rate"],
            "num_leaves": self.cfg["num_leaves"],
            "min_data_in_leaf": self.cfg["min_data_in_leaf"],
            "bagging_fraction": self.cfg["bagging_fraction"],
            "feature_fraction": self.cfg["feature_fraction"],
            "min_child_samples": self.cfg["min_child_samples"],
            "max_depth": -1,
            "seed": 10
        }

        self.evals_result = {}
        self.model = lgbt.train(
            params=lgbm_params,
            train_set=train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            num_boost_round=max_rounds,  
            callbacks=[
                lgbt.early_stopping(stopping_rounds=10),
                lgbt.log_evaluation(10),
                lgbt.record_evaluation(self.evals_result),
            ],
            feval=f1_eval,
        )

    def eval(self, rounds=100):
        self.train(rounds)

        print(len(self.X_test))

        predict_arr = self.model.predict(self.X_test)
        predict_arr = np.argmax(predict_arr, axis=1)

        nclasses = len(np.unique(self.y_test))

        f1 = f1_score(self.y_test, predict_arr, average='macro')

        self.week_evals.append(f1)

        eval_arr = [[index, value] for index, value in enumerate(self.week_evals)]

        return f1, eval_arr, self.y_test, predict_arr, [str(i) for i in range(nclasses)]

    def next_period(self):
        print(self.week_i)
        self.day_i += 1
        
        if self.period == "week":
            self.day_i += 6
        
        if self.day_i % 7 == 0:
            self.week_i += 1


        if self.week_i >= len(self.week_arr):
            return False

        try:
            dataset = CESNET_TLS_Year22(data_root="/storage/brno2/home/sigull/datasets/CESNET-TLS-Year22/", size="L")
        except:
            dataset = CESNET_TLS_Year22(data_root="~/datasets/CESNET-TLS-Year22/", size="XS")

        if self.period == "week":
            common_params = {
                "dataset" : dataset,
                "apps_selection" : AppSelection.ALL_KNOWN,
                "train_period_name": self.week_arr[self.week_i],
                "train_dates": self.day_arr[self.day_i-6:self.day_i],
                "need_test_set": False,
            }

        else:
            common_params = {
                "dataset" : dataset,
                "apps_selection" : AppSelection.ALL_KNOWN,
                "train_period_name": self.week_arr[self.week_i],
                "train_dates": {self.day_arr[self.day_i]},
                "need_test_set": False,
            }

        dataset_config = DatasetConfig(**common_params)
        dataset.set_dataset_config_and_initialize(dataset_config)
        train_dataframe = dataset.get_train_df(flatten_ppi=True)
        val_dataframe = dataset.get_val_df(flatten_ppi=True)

        known_apps = dataset.get_known_apps()
        local_to_global = {i: self.label_to_global_y[app] for i, app in enumerate(known_apps)}

        self.prev_Xs.append(np.copy(self.X[self.chosen_indices]))
        self.prev_ys.append(np.copy(self.y[self.chosen_indices]))
        self.prev_valXs.append(np.copy(self.X_val))
        self.prev_valys.append(np.copy(self.y_val))

        self.X = train_dataframe.drop(columns="APP").to_numpy()
        self.y = train_dataframe["APP"].to_numpy()
        self.y = np.array([local_to_global[y] for y in self.y])

        self.X_val = val_dataframe.drop(columns="APP").to_numpy()
        self.y_val = val_dataframe["APP"].to_numpy()
        self.y_val = np.array([local_to_global[y] for y in self.y_val])

        self.chosen_indices = np.array([0])
        self.unknown_indices = np.array([i for i in range(1, len(self.X))])

        return True

class Uncertainty_Sweep(Sweep_Class):
    def run(self, batch_size=1):
        day = 0
        while True:
            # print("Current day:", day)
            # print("Current dataset len:", self.get_current_train_dataset_len())

            while(len(self.unknown_indices) >= batch_size):
                self.train(20)
                predict_arr = self.model.predict(self.X[self.unknown_indices])
                average_unc = 1 - (np.sum(np.max(predict_arr, axis=1)) / len(predict_arr))
                # print(average_unc)
                if self.cfg["average_unc_cutoff"] > average_unc:
                    break

                sorted_indexes_1toN = np.argsort(np.max(predict_arr, axis=1))[:batch_size]
                selected_indexes = self.unknown_indices[sorted_indexes_1toN]
                
                self.chosen_indices = np.concatenate(([self.chosen_indices, np.array(selected_indexes)]))
                self.unknown_indices = np.delete(self.unknown_indices, sorted_indexes_1toN) 
            
            f1 = self.eval(20)
            print("f1 score:", end=" ")
            print(f1)

            # day += 1

            if not self.next_period():
                break

class Random_Sweep(Sweep_Class):
    def run(self):
        self.period = "week"
        day = 0
        while True:
            print(day)
            self.unknown_indices = np.random.permutation(self.unknown_indices)
            self.chosen_indices = np.concatenate(([self.chosen_indices, self.unknown_indices[:int(len(self.X) * self.cfg["random_frac"])]]));            
            self.unknown_indices = self.unknown_indices[int(len(self.X) * self.cfg["random_frac"]):]
            
            day += 7

            f1 = self.eval(20)
            print("f1 score:", end=" ")
            print(f1)

            if not self.next_period():
                break

def depth(el):
    d = 0
    try:
        d = depth(el['left_child']) + 1
    except:
        pass

    try:
        d_new = depth(el['left_child']) + 1
        if d_new > d:
            d = d_new
    except:
        pass

    return d


class LAL_Sweep(Sweep_Class):
    def get_pred(self, iters):
        predict_arr = self.model.predict(
            self.X[self.unknown_indices], start_iteration=iters, num_iteration=1, raw_score=True
        )

        return np.max(predict_arr, axis=1)

    def run(self):
        train_iters = 20

        while True:
            self.train(train_iters)
            val_log_loss = self.evals_result["val"]["multi_logloss"][-1]

            # print(self.model.model_to_string())
            tree_heights = []
            model_dump = self.model.dump_model()
            for tree in model_dump['tree_info']:
                height = 0
                try:
                    data = tree['tree_structure']['left_child']
                    while True:
                        height += 1
                        data = data["left_child"]
                except:
                    pass
                
                if height != 0:
                    tree_heights.append(height)

            average_height = np.mean(tree_heights)
            print(average_height)

            n_labelled = np.size(self.chosen_indices)
            n_dim = np.shape(self.X)[1]

            temp = np.array([self.get_pred(index) for index in range(1, train_iters)])
            f_1 = np.mean(temp, axis=0)
            f_2 = np.std(temp, axis=0)
            # - proportion of positive points
            f_3 = (1) * np.ones_like(f_1)
            # originally out of bag estimate. Adapted to val log loss
            f_4 = val_log_loss * np.ones_like(f_1)
            # - coeficient of variance of feature importance
            f_5 = np.std(self.model.feature_importance()/n_dim) * np.ones_like(f_1)
            # - estimate variance of forest by looking at average of variance of some predictions
            f_6 = np.mean(f_2, axis=0) * np.ones_like(f_1)
            # - compute the average depth of the trees in the forest
            f_7 = average_height * np.ones_like(f_1)
            # - number of already labelled datapoints
            f_8 = np.size(self.chosen_indices) * np.ones_like(f_1)

            LALfeatures = np.concatenate(([f_1], [f_2], [f_3], [f_4], [f_5], [f_6], [f_7], [f_8]), axis=0)
            LALfeatures = np.transpose(LALfeatures)

            if not self.next_period():
                break

