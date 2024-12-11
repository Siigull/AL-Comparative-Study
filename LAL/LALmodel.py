import numpy as np
import math
import lightgbm as lgbt

class LALmodel:

    def __init__(self, all_data_for_lal, all_labels_for_lal):
        
        self.all_data_for_lal = all_data_for_lal
        self.all_labels_for_lal = all_labels_for_lal
        
    def crossValidateLALmodel(self):
            
        possible_estimators = [500, 1000, 5000]
        possible_depth = [5, 10, 20]
        possible_features =[3, 5, 7]
        small_number = 0.0001
    
        best_score = -math.inf

        self.best_est = 0
        self.best_depth = 0
        self.best_feat = 0
    
        print('start cross-validating..')
        for est in possible_estimators:
            for depth in possible_depth:
                for feat in possible_features:
                    model = lgbt.LGBMRegressor(n_estimators = est, max_depth=depth, max_features=feat, n_jobs=-1)
                    new_model = model.fit(self.all_data_for_lal[:,:-1], np.ravel(self.all_labels_for_lal), eval_set=[(self.all_data_for_lal[:,:-1], np.ravel(self.all_labels_for_lal))])
                    train_log_loss = new_model.evals_result_["valid_0"]["l2"][-1]
                    if train_log_loss>best_score+small_number:
                        self.best_est = est
                        self.best_depth = depth
                        self.best_feat = feat
                        self.model = model
                        best_score = train_log_loss
        # now train with the best parameters
        print('best parameters = ', self.best_est, ', ', self.best_depth, ', ', self.best_feat, ', with the best score = ', best_score)
        return best_score
