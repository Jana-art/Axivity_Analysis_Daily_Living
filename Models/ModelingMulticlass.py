from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from Models.Modeling import Models

from constants import __PARAM_GRID__

import pandas as pd
import numpy as np


class ModelsMulticlass(Models):


    def __init__(self, filteredFeaturesData, labels):

        super().__init__(filteredFeaturesData, labels)

        self.f1_scores = {}
        self.precision_scores = {}
        self.recall_scores = {}

        # For later - misclassification analysis
        self.FH_dict = defaultdict(int)
        self.FL_dict = defaultdict(int)
        self.T_dict = defaultdict(int)

    def train_models(self, seed=0, thres=0.5, model="RF", verbose=False, saveModel=False):
        return super().train_models(seed, thres, model, verbose, saveModel)

    def save_model_scores(self, model, clf, X_train, y_train, cv):

        self.f1_scores[model] = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_weighted").mean()
        self.precision_scores[model] = cross_val_score(clf, X_train, y_train, cv=cv, scoring="precision_weighted").mean()
        self.recall_scores[model] = cross_val_score(clf, X_train, y_train, cv=cv, scoring="recall_weighted").mean()


    def iterate_models(self, model_name, seed):

        clf = None

        if model_name == "RFC":
            clf = GridSearchCV(RandomForestClassifier(), __PARAM_GRID__[model_name], scoring="f1_weighted")
            # clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed)
        elif model_name == "SVM":
            clf = GridSearchCV(SVC(), __PARAM_GRID__[model_name], scoring="f1_weighted")
            # clf = svm_linear = SVC(C=1.0, kernel='linear', random_state=seed)
        elif model_name == "NB":
            clf = GridSearchCV(GaussianNB(), __PARAM_GRID__[model_name], scoring="f1_weighted")
            # clf = GaussianNB()
        elif model_name == "LOGR":
            clf = GridSearchCV(LogisticRegression(), __PARAM_GRID__[model_name], scoring="f1_weighted")
            # clf = LogisticRegression(solver="lbfgs", penalty="l2", class_weight="balanced", random_state=seed)

        return clf


    def display_scores(self, model):
        print("f1 score: " + str(self.f1_scores[model]) + "\nrecall: " + str(self.recall_scores[model]) + "\nprecision:" + str(
            self.precision_scores[model]))


    def select_best_model(self):
        chosen_model = list(self.f1_scores.keys())[list(self.f1_scores.values()).index(np.max(list(self.f1_scores.values())))]
        return chosen_model


    def get_results_scores(self, pred, X_test, y_test, chosen_model, clf_chosen, verbose):

        chose_model_f1_score = f1_score(y_test, pred, average="weighted")
        chose_model_precision_score = precision_score(y_test, pred, average="weighted")
        chose_model_recall_score = recall_score(y_test, pred, average="weighted")
        Cmat = confusion_matrix(y_test, pred)

        self.update_misclassifications(y_test, pred, verbose)

        if verbose:
            print("Chosen model is: " + chosen_model)
            print("the f1 score of the model on the test set is: " + str(chose_model_f1_score))
            print("the recall score of the model on the test set is: " + str(chose_model_recall_score))
            print("the precision score of the model on the test set is: " + str(chose_model_precision_score))
            print("the confusion matrix of the model on the test set is: "+ "\n" + str(Cmat))

        return [chose_model_f1_score, chose_model_precision_score, chose_model_recall_score]


    def update_misclassifications(self, y_test, pred, verbose):

        results = pd.DataFrame([y_test.index.values, y_test.values, pred]).transpose()
        results.columns = ["SubjectID","y_test","pred"]
        FH = list(results[results["y_test"] < results["pred"]]["SubjectID"]) # false - higher (than true label)
        FL = list(results[results["y_test"] > results["pred"]]["SubjectID"]) # false - lower (than true label)
        T = list(results[results["y_test"] == results["pred"]]["SubjectID"]) # true prediction

        for s in FH:
            self.FH_dict[s] = self.FH_dict[s] + 1
        for j in FL:
            self.FL_dict[j] = self.FL_dict[j] + 1
        for k in T:
            self.T_dict[k] = self.T_dict[k] + 1

        if verbose:
            self.get_misclassifications()


    def get_misclassifications(self):

        print("False samples - predicted higher than label (relevant only for ordinal labels): " + str(self.FH_dict))
        print("False samples - predicted lower than label (relevant only for ordinal labels): " + str(self.FL_dict))
        print("True samples: " + str(self.T_dict))

        return {"FH_dict":self.FH_dict, "FL_dict":self.FL_dict, "T_dict":self.T_dict}

    def get_prob_score(self, X_test, y_test, chosen_model, clf_chosen):

        if(chosen_model == "SVM"):
            return clf_chosen.decision_function(X_test).transpose()
        else:
            return clf_chosen.predict_proba(X_test)[:,1]

    def get_test_results_scores(self, y_true, pred):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Confusion matrix:")
        print(confusion_matrix(y_true, pred))
        return f1_score(y_true, pred, average="weighted")