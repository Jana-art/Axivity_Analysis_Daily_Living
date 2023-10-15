from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_score, GroupShuffleSplit
import messages
from constants import __MODELS__, __FOLDS_NUM__, __MODELING_MODE__, __TEST_SIZE__,__LABELING_METHOD__,__NUMBER_OF_FEATURES_MUTUAL_INFO__,__TEST_SCORES_ITERATIONS__,__SEED__
from constants import __NESTED_FLAG__
from datetime import datetime
import pickle
import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class Models():

    def __init__(self, filteredFeaturesData, labels):

        self.X = filteredFeaturesData
        self.y = labels
        self.clf_chosen = -1

    """
    Training models on the data with the following steps:
    1. Split the data to two sets: (A) train/validation set (B) test set.
    2. Apply cross validation to set (A) for model selection
    3. Choose the best algorithm by the maximal score received on KFold cross validation
    4. Apply the chosen algorithm on the test set from step (1) to evaluate its actual performance on new data
    """
    def train_models(self, seed, thres = 0.5, model = "RF", verbose = False, saveModel = True):
        saveModel = True
        cv = StratifiedKFold(n_splits=__FOLDS_NUM__, shuffle=True, random_state=seed)
        models = {}
        X = self.X
        y = self.y

        # keep aside a test set to estimate the performance of the chosen model
        # X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = __TEST_SIZE__, shuffle=True, random_state = seed) #TODO change seed

        # Perform model selection using cross validation:
        Subjects = [x[0:-3] for x in X.index]
        Subject_groups = pd.Series(Subjects, index=X.index)
        Subjects = pd.DataFrame(data=Subjects, index=X.index, columns=["groups"])
        groups = Subjects.groupby("groups")

        gss = GroupShuffleSplit(n_splits=5, train_size=0.8,random_state=np.random.randint(0, 100))  # UnNested - groupshuffle to seperate visits 08/10/2023

        print("Starting Cross-Validation: split data to train test")

        #gss_outer = GroupShuffleSplit(n_splits=5, train_size=0.8,random_state=np.random.randint(0, 100))  # nested - groupshuffle to seperate visits 08/10/2023



        for fold, (train_idx, test_idx) in enumerate(gss.split(X, y, Subject_groups)):


            print(f"Fold {fold}")

            X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

            for model in __MODELS__[__MODELING_MODE__]:

                clf = self.iterate_models(model, __SEED__)


                # Nested Cross-Validaion 08/10/2023

                # print("TRAIN:", train_idx, "TEST:", test_idx)


                clf = self.save_model_scores(model, clf, X_train, y_train, X_test, y_test, cv) # Non nested cross validation and results

                print(model)
                models[model] = clf  # .best_estimator_




        # Select the model that got the best scores:
        chosen_model = self.select_best_model()

        # test the chosen model on the test set:
        clf_chosen = models[chosen_model]

        model_object = clf_chosen.best_estimator_
        model_object.fit(X, y) #TODO take all train data, fit best model

        self.chosen_model_names.append(model_object.__class__.__name__)
        self.chosen_model_params.append(model_object.get_params())


        if saveModel: # TODO:POC

            self.clf_chosen = model_object

            for i in range(0,__TEST_SCORES_ITERATIONS__):
                n = i
                if os.path.isfile(os.path.join(r"*PATH*_ \Data\data\debug_files",  "ModelObject" + __LABELING_METHOD__ + str(n)+ '.p')):
                    continue
                else:
                    with open(os.path.join(
                            r"*PATH*_ \Data\data\debug_files",
                            "ModelObject" + __LABELING_METHOD__ + str(n) + '.p'),
                              'wb') as outputFile:
                        pickle.dump(model_object, outputFile)
                    break


        return model_object, chosen_model

        # return res_scores

    def get_CV_results(self,chosen_model,i):


        return

    def display_scores(self,model):
        return

    def save_model_scores(self, model, clf, X_train, y_train,X_test,y_test,cv):

        return

    def get_all_results(self, chosen_model, N_train, N_test):

        return
    def get_results_scores_v2(self,chosen_model):
        return

    def iterate_models(self, model, seed):
        return

    def select_best_model(self):
        return

    def get_results_scores(self, pred, X_test, y_test, chosen_model, clf_chosen, verbose):
        return

    def get_test_results_scores(self, y_true, pred):
        return

    def save_model(self, clf):
        self.clf_chosen = clf

    def get_model(self):
        if self.clf_chosen != -1:
            return self.clf_chosen
        else:
            messages.write_message_no_model()

    def predict_from_model(self, X):
        if self.clf_chosen != -1:
            return self.clf_chosen.predict(X)
        else:
            messages.write_message_no_model()

