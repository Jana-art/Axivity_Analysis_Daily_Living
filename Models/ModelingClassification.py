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


class ModelsClassification(Models):


    def __init__(self, filteredFeaturesData, labels):

        super().__init__(filteredFeaturesData, labels)

        self.f1_scores = {}
        self.precision_scores = {}
        self.recall_scores = {}

        # For later - misclassification analysis
        self.FP_dict = defaultdict(int)
        self.FN_dict = defaultdict(int)
        self.TN_dict = defaultdict(int)
        self.TP_dict = defaultdict(int)

    def train_models(self, seed=0, thres=0.5, model="RF", verbose=False, saveModel=False):
        return super().train_models(seed, thres, model, verbose, saveModel)

    def save_model_scores(self, model, clf, X_train, y_train, cv):

        self.f1_scores[model] = clf.fit(X_train, y_train).best_score_
        # self.f1_scores[model] = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1").mean()
        # self.precision_scores[model] = cross_val_score(clf, X_train, y_train, cv=cv, scoring="precision").mean()
        # self.recall_scores[model] = cross_val_score(clf, X_train, y_train, cv=cv, scoring="recall").mean()


    def iterate_models(self, model_name, seed):

        clf = None

        if model_name == "RFC":
            clf = GridSearchCV(RandomForestClassifier(), __PARAM_GRID__[model_name], scoring="f1")
            # clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed)
        elif model_name == "SVM":
            clf = GridSearchCV(SVC(), __PARAM_GRID__[model_name], scoring="f1")
            # clf = svm_linear = SVC(C=1.0, kernel='linear', random_state=seed)
        elif model_name == "NB":
            clf = GridSearchCV(GaussianNB(), __PARAM_GRID__[model_name], scoring="f1")
            # clf = GaussianNB()
        elif model_name == "LOGR":
            clf = GridSearchCV(LogisticRegression(), __PARAM_GRID__[model_name], scoring="f1")
            # clf = LogisticRegression(solver="lbfgs", penalty="l2", class_weight="balanced", random_state=seed)

        return clf


    def display_scores(self, model):
        print("f1 score: " + str(self.f1_scores[model]) + "\nrecall: " + str(self.recall_scores[model]) + "\nprecision:" + str(
            self.precision_scores[model]))


    def select_best_model(self):
        chosen_model = list(self.f1_scores.keys())[list(self.f1_scores.values()).index(np.max(list(self.f1_scores.values())))]
        return chosen_model


    def get_results_scores(self, pred, X_test, y_test, chosen_model, clf_chosen, verbose):

        chose_model_f1_score = f1_score(y_test, pred)
        chose_model_precision_score = precision_score(y_test, pred)
        chose_model_recall_score = recall_score(y_test, pred)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        chose_model_specificity_score = tn / (tn + fp)

        auc_score = -1
        prob = self.get_prob_score(X_test, y_test, chosen_model, clf_chosen)
        auc_score = roc_auc_score(y_test, prob)
        fpr, tpr, thresh = roc_curve(y_test, prob)

        # scores with optimal threshold according to ROC-curve:
        opt_thres =thresh[np.argmax(tpr - fpr)]
        pred = prob > opt_thres
        chose_model_f1_score = f1_score(y_test, pred)
        chose_model_precision_score = precision_score(y_test, pred)
        chose_model_recall_score = recall_score(y_test, pred)

        # plt.plot(fpr, tpr)
        # plt.show()

        self.update_misclassifications(y_test, pred, verbose)

        if verbose:
            print("Chosen model is: " + chosen_model)
            print("the f1 score of the model on the test set is: " + str(chose_model_f1_score))
            print("the recall score of the model on the test set is: " + str(chose_model_recall_score))
            print("the precision score of the model on the test set is: " + str(chose_model_precision_score))
            print("the specificity score of the model on the test set is: " + str(chose_model_specificity_score))
            print("the auc score of the model on the test set is: " + str(auc_score))

        return [chose_model_f1_score, chose_model_precision_score, chose_model_recall_score, chose_model_specificity_score, auc_score]


    def update_misclassifications(self, y_test, pred, verbose):

        results = pd.DataFrame([y_test.index.values, y_test.values, pred]).transpose()
        results.columns = ["SubjectID","y_test","pred"]
        FP = list(results[(results["y_test"] == 0) & (results["pred"] == 1)]["SubjectID"])
        FN = list(results[(results["y_test"] == 1) & (results["pred"] == 0)]["SubjectID"])
        TN = list(results[(results["y_test"] == 0) & (results["pred"] == 0)]["SubjectID"])
        TP = list(results[(results["y_test"] == 1) & (results["pred"] == 1)]["SubjectID"])

        for s in FP:
            self.FP_dict[s] = self.FP_dict[s] + 1
        for j in FN:
            self.FN_dict[j] = self.FN_dict[j] + 1
        for k in TN:
            self.TN_dict[k] = self.TN_dict[k] + 1
        for n in TP:
            self.TP_dict[n] = self.TP_dict[n] + 1

        if verbose:
            self.get_misclassifications()

    def get_misclassifications(self):

        print("FP samples: " + str(self.FP_dict))
        print("FN samples: " + str(self.FN_dict))
        print("TP samples: " + str(self.TP_dict))
        print("TN samples: " + str(self.TN_dict))

        return {"FP_dict":self.FP_dict, "FN_dict":self.FN_dict, "TP_dict":self.TP_dict,"TN_dict":self.TN_dict}

    def get_prob_score(self, X_test, y_test, chosen_model, clf_chosen):

        if(chosen_model == "SVM"):
            return clf_chosen.decision_function(X_test).transpose()
        else:
            return clf_chosen.predict_proba(X_test)[:,1]

    def get_test_results_scores(self, y_true, pred):
        return f1_score(y_true, pred)
