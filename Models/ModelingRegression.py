
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge,SGDRegressor,Lars,ElasticNet
from sklearn.svm import SVR

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from matplotlib import pyplot as plt
from Models.Modeling import Models
import numpy as np
from sklearn.metrics import mean_absolute_error
import xgboost
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

import os
from constants import __PARAM_GRID__,__VERBOSE__,__LABELING_METHOD__,__DATE__,__TEST_SCORES_ITERATIONS__

class ModelsRegression(Models):


    def __init__(self, filteredFeaturesData, labels):

        super().__init__(filteredFeaturesData, labels)
        self.mean_absolute_error = {}
        self.R_2 = {}
        self.res_scores = list()
        self.R_2_folds = {} # train-test CV, in-fold test scores
        self.MAE = {}
        self.Pearson_corr = {}
        self.Pearson_p = {}
        self.N_train = {}
        self.N_test = {}
        self.Mean_pearson = []
        self.Mean_R_CV = []
        self.Mean_R2 = []
        self.Mean_MAE = []
        self.std_pearson = {}
        self.std_R_CV = {}
        self.std_R2 = {}
        self.std_MAE = {}
        self.chosen_model_names = []
        self.chosen_model_params = []
        # self.input_path = input_path

    def train_models(self, seed=0, thres = 0.5, model = "RF", verbose = False, saveModel = False):

        return super().train_models(seed, thres, model, verbose, saveModel)

    def get_all_results(self,chosen_model,N_train,N_test):

        R_2 = [x for x in self.Mean_R2]
        R_2_CV = [x for x in self.Mean_R_CV]
        MAE = [x for x in self.Mean_MAE]
        Pearson_corr = [x for x in self.Mean_pearson]

        Mean_R2 = np.mean(R_2)
        Mean_R_CV = np.mean(R_2_CV)
        Mean_MAE = np.mean(MAE)
        Mean_pearson = np.mean(Pearson_corr)

        std_pearson = np.std(Pearson_corr)
        std_R_CV = np.std(R_2_CV)
        std_R2 = np.std(R_2)
        std_MAE = np.std(MAE)

        result_dict = {'Model names': self.chosen_model_names,'Model parameters': self.chosen_model_params, 'Pearson_statistic': Pearson_corr,
                       'MAE': MAE, 'R2': R_2, 'R2_CV': R_2_CV, 'N_train': N_train, 'N_test': N_test, 'Mean Pearson': Mean_pearson,
                       'Mean R_CV': Mean_R_CV, 'Mean R2': Mean_R2, 'Mean MAE': Mean_MAE,'std Pearson': std_pearson,
                       'std R_CV': std_R_CV, 'std R2': std_R2, 'std MAE': std_MAE}

        for key in result_dict:
            N = len([result_dict[key]])
            if key == "Model names" or key == "Model parameters":
                continue

            elif N == 1:
                result_dict[key]= np.round(result_dict[key], 2)

            else:    # rounding to K using round()
                i=0
                N=len([result_dict[key]])
                for kk in range(0,N):
                    result_dict[key][kk] = np.round(result_dict[key][kk], 2)

        result_df = pd.DataFrame(result_dict)

        result_df.to_csv(os.path.join(r"*PATH*_ \Results", "CVResults" + __LABELING_METHOD__ + __DATE__ + '.csv'))

        return

    def get_CV_results(self,chosen_model,i):

        Model_Names = [x for x in self.Pearson_corr.keys()]
        Pearson_corr = [x for x in self.Pearson_corr.values()]
        Pearson_p = [x for x in self.Pearson_p.values()]
        Pearson_p_significancy = [x for x in np.asarray(Pearson_p) < 0.05]

        if not np.asarray(Pearson_p_significancy).all():
            Pearson_p_significancy = 0
        else:
            Pearson_p_significancy = 1

        R_2 = [x for x in self.R_2.values()]
        R_2_CV = [x for x in self.R_2_folds.values()]
        MAE = [x for x in self.MAE.values()]
        N_train = [x for x in self.N_train.values()]
        N_test = [x for x in self.N_test.values()]

        Mean_pearson_fold = np.mean(Pearson_corr,axis=1)
        Mean_R_CV_fold = np.mean(R_2_CV,axis=1)
        Mean_R2_fold = np.mean(R_2,axis=1)
        Mean_MAE_fold = np.mean(MAE,axis=1)


        Mean_pearson = np.mean(self.Pearson_corr[chosen_model])
        Mean_R_CV = np.mean(self.R_2_folds[chosen_model])
        Mean_R2 = np.mean(self.R_2[chosen_model])
        Mean_MAE = np.mean(self.MAE[chosen_model])

        std_pearson = np.std(Pearson_corr,axis=1)
        std_R_CV = np.std(R_2_CV,axis=1)
        std_R2 = np.std(R_2,axis=1)
        std_MAE = np.std(MAE,axis=1)

        self.Mean_pearson.append(Mean_pearson)
        self.Mean_R_CV.append(Mean_R_CV)
        self.Mean_R2.append(Mean_R2)
        self.Mean_MAE.append(Mean_MAE)


        result_dict = {'Model names': Model_Names,'Pearson_statistic':  Pearson_corr , 'Pearson_p_value_total': Pearson_p,
                       'MAE': MAE, 'R2': R_2 , 'R2_CV':R_2_CV,'N_train':N_train,'N_test':N_test,'Pearson p-value significancy': Pearson_p_significancy,'Mean Pearson': Mean_pearson_fold,'Mean R_CV': Mean_R_CV_fold,'Mean R2':Mean_R2_fold,'Mean MAE': Mean_MAE_fold,'std Pearson': std_pearson,
                       'std R_CV': std_R_CV, 'std R2': std_R2, 'std MAE': std_MAE}

        for key in result_dict:
            if key == "Model names" or key == 'Pearson p-value significancy':
                continue
            else:    # rounding to K using round()

                for jj in range(0,len(result_dict['Model names'])):
                    result_dict[key][jj] = np.round(result_dict[key][jj], 2)

        result_df = pd.DataFrame(data=result_dict)

        result_df.to_csv(os.path.join(r"*PATH*_ \Results","FoldResults" + str(i) + __LABELING_METHOD__ + __DATE__ + '.csv'))


        return result_df


    def save_model_scores(self, model, clf, X_train, y_train,X_test,y_test, cv):

        # self.mean_absolute_error[model] = clf.fit(X_train, y_train).best_score_
        # self.R_2[model] = cross_val_score(clf, X_train, y_train, cv=cv, scoring="r2").mean()
        # Group folds append

        R2 = clf.fit(X_train, y_train).best_score_
        y_pred = clf.best_estimator_.predict(X_test)
        R_2_folds = r2_score(y_test, y_pred) #Added score from in-fold test set

        MAE = mean_absolute_error(y_pred,y_test)
        Pearson = pearsonr(y_pred,y_test)
        Pearson_corr = Pearson[0]
        Pearson_p = Pearson[1]
        N_train = len(y_train)
        N_test = len(y_test)

        # Added 08/02/2023
        # R2 = clf.predict(X_test, y_test).best_score_

        if model in self.N_train:
            self.N_train[model] = np.append(self.N_train[model], N_train)
        else:
            self.N_train.__setitem__(model, N_train)

        if model in self.N_test:
            self.N_test[model] = np.append(self.N_test[model], N_test)
        else:
            self.N_test.__setitem__(model, N_test)

        if model in self.R_2:
            self.R_2[model] = np.append(self.R_2[model], R2)
        else:
            self.R_2.__setitem__(model, R2)

        if model in self.R_2_folds:
            self.R_2_folds[model] = np.append(self.R_2_folds[model], R_2_folds)
        else:
            self.R_2_folds.__setitem__(model, R_2_folds)

        if model in self.Pearson_corr:
            self.Pearson_corr[model] = np.append(self.Pearson_corr[model], Pearson_corr)
        else:
            self.Pearson_corr.__setitem__(model, Pearson_corr)

        if model in self.Pearson_p:
            self.Pearson_p[model] = np.append(self.Pearson_p[model], Pearson_p)
        else:
            self.Pearson_p.__setitem__(model, Pearson_p)

        if model in self.MAE:
            self.MAE[model] = np.append(self.MAE[model], MAE)
        else:
            self.MAE.__setitem__(model, MAE)

        return clf

    def iterate_models(self, model_name, seed):

        clf = None
        vv = 1
        if model_name == "RFR":
            clf = GridSearchCV(RandomForestRegressor(), __PARAM_GRID__[model_name], scoring="r2",verbose=vv)

        elif model_name == "LINR":
            clf = GridSearchCV(LinearRegression(), __PARAM_GRID__[model_name], scoring="r2")

        elif model_name == "RDG":
            clf = GridSearchCV(Ridge(), __PARAM_GRID__[model_name], scoring="r2",verbose=vv)

        elif model_name == "LASSO": #ToDo Uncomment for LASSO
            clf = GridSearchCV(Lasso(), __PARAM_GRID__[model_name], scoring="r2",verbose=vv)

        elif model_name == "XGB":
            clf = GridSearchCV(XGBRegressor(), __PARAM_GRID__[model_name], scoring="r2",verbose=vv)

        elif model_name == "SVR":
            clf = GridSearchCV(SVR(), __PARAM_GRID__[model_name], scoring="r2", verbose=vv)

        elif model_name == "LOGR":
            clf = GridSearchCV(LogisticRegression(), __PARAM_GRID__[model_name], scoring="r2", verbose=vv)

        elif model_name == "BR":
            clf = GridSearchCV(BayesianRidge(),__PARAM_GRID__[model_name], scoring="r2", verbose=vv)

        elif model_name == "SGD":
            clf = GridSearchCV(SGDRegressor(),__PARAM_GRID__[model_name], scoring="r2", verbose=vv)

        elif model_name == 'Lars':
            clf = GridSearchCV(Lars(),__PARAM_GRID__[model_name], scoring="r2", verbose=vv)

        elif model_name == 'EN':
            clf = GridSearchCV(ElasticNet(), __PARAM_GRID__[model_name], scoring="r2", verbose=vv)

        elif model_name == 'ADB':
            clf = GridSearchCV(AdaBoostRegressor(), __PARAM_GRID__[model_name], scoring="r2", verbose=vv)

        return clf

    def select_best_model(self):


        chosen_model = list(self.R_2_folds.keys())[np.argmax(np.mean(list(self.R_2_folds.values()), axis=1))]

        # Reset

        print("Chosen Model:" + str(chosen_model))
        return chosen_model

    def display_scores(self, model):

        print("mean_absolute_error: " + str(self.mean_absolute_error[model]))
        print("R_2: " + str(self.R_2[model]))

    def get_results_scores_v2(self,chosen_model):

        corr = np.mean(list(self.Pearson_corr[chosen_model]))
        MAE = np.mean(list(self.MAE[chosen_model]))
        Pval = np.mean(list(self.Pearson_p[chosen_model]))
        R_2 = np.mean(list(self.R_2_folds[chosen_model]))


        return corr,MAE,Pval,R_2

    def get_results_scores(self, pred, X_test, y_test, chosen_model, clf_chosen, verbose):

        corr = pearsonr(pred, y_test)
        MAE = mean_absolute_error(pred, y_test)
        # R_2 = r2_score(pred, y_test)
        R_2 = r2_score(y_test, pred)

        verbose = True

        plot = True
        if plot:

            plt.show()

            fig, ax = plt.subplots(figsize=(6.4, 4.8))
          

            # ax.text(0.1, 0.9, 'text', size=15, color='purple')
                              
            ax.scatter(y_test, pred,zorder=1,alpha=0.6,color='#0000CC')

            m, b = np.polyfit(y_test, pred, 1)

            ax.plot(y_test, m * y_test + b, 'k')

            # plt.plot(y_val, yhat, color='k')
            if corr[1] < 0.001:
                ax.text(0.82 * max(y_test), 0.4 * max(pred),
                        'Pearson R = ' + str(np.around(corr[0], 2)) + '\n' + 'Pvalue < 0.001', fontsize=14,
                        bbox=dict(facecolor='None', alpha=0.5))
            else:
                ax.text(0.82 * max(y_test), 0.4 * max(pred), 'Pearson R = ' + str(np.around(corr[0], 2)) + '\n' + 'Pvalue = ' + str(
                    np.around(corr[1], 2)), fontsize=14, bbox=dict(facecolor='None', alpha=0.5))

            # plt.plot(y_test, y_test)
            
            if __LABELING_METHOD__== "UPDRSI":
                label_name ='MDS-UPDRS I'
            elif __LABELING_METHOD__ == "UPDRSII":
                label_name = 'MDS-UPDRS II'
            elif __LABELING_METHOD__ == "UPDRSIII":
                label_name = 'MDS-UPDRS III'
            elif __LABELING_METHOD__ == "PIGD_score":
                label_name = 'PIGD score'
            elif __LABELING_METHOD__ == "TD_score":
                label_name = 'TD score'

            plt.suptitle("Regression model, label: " + label_name, fontsize=25)

            # plt.title("Correlation: " + str(sl_corr),fontsize=8)
            plt.xlabel("Actual " + label_name, fontsize=20)
            plt.ylabel("Predicted " + label_name, fontsize=20)

            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            fig.set_size_inches((12.4, 8.8), forward=False)

            for i in range(0,__TEST_SCORES_ITERATIONS__):
                n = i
                if os.path.isfile(os.path.join(r"*PATH*_ \Results","train_" + label_name + str(n) + ".png")):

                    continue
                else:
                    plt.savefig(os.path.join(r"*PATH*_ \Results","train_" + label_name + str(n) + ".png"), dpi=1000)
                    break



    
            
        if verbose:
            print("Chosen model is: " + chosen_model)
            print("CV test size: " + str(len(y_test)))

            print("the pearson correlation score of the model on the CV test set is: " + str(corr))
            print("the pearson MAE score of the model on the CV test set is: " + str(MAE))
            print("the R squared score of the model on the CV test set is: " + str(R_2))




        return corr,MAE,R_2

    def get_test_results_scores(self,y_true, pred):
        return pearsonr(y_true, pred)