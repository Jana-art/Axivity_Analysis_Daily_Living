# Copyright [2024] [Center for the Study of Movement, Cognition, and Mobility. Tel-Aviv Sourasky Medical Center, Tel Aviv, Israel]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, Ridge
from itertools import compress
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.feature_selection._sequential import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import f1_score, mean_absolute_error
from constants import __DATE__, __FILTERS__  , __ALL_FEATURE_NAMES_LIST__ , __ALL_FEATURE_NAMES_LIST_WITH_ENGINEERED__, __LOW_VARIANCE_FEATURES_TRES__, __MAX_NUM_OF_FEATURES_AFTER_UNIVARIATE_FILTERING__, __ENGINEERED_FEATURES__, __FRAGMENTATION_FEATURE_NAMES__, __FOLDS_NUM__, __LABELING_METHOD__, __GAIT_QUALITY_FEATURES__, __GAIT_QUANTITY_FEATURES__, __TRANSITIONS_FEATURES__, __SLEEP_FEATURE_NAMES_LIST__, __UNIVARIATE_TEST_DICT__, __MODELING_MODE__, __NUMBER_OF_FEATURES_FORWARD_SELECTION__,__FEATURE_TYPE_DICT__,__NUMBER_OF_FEATURES_MUTUAL_INFO__,__NUMBER_OF_FEATURES_MRMR__,\
    __LOAD_FEATURES_FLAG__
import numpy as np
import os
from matplotlib import pyplot as plt
import itertools
import random
import pickle
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.svm import OneClassSVM
from scipy import io


class FeaturesSelector():


    def __init__(self, processedData, labels, input_path,model_names,loaded_features,features_score_thres,k):

        self.data = processedData
        self.labels = labels
        self.featureNames = list(set(__ALL_FEATURE_NAMES_LIST_WITH_ENGINEERED__).intersection(set(self.data.columns)))
        self.filteredFeatures = self.featureNames
        self.initial_amount_of_features = len(self.featureNames)
        self.input_path = input_path
        self.model_names = model_names
        self.current_model = []
        self.load_flag = __LOAD_FEATURES_FLAG__
        self.loaded_features = loaded_features
        self.k_features= k
        self.feature_thresh = features_score_thres
        
    def select_features(self, subset=[]):

        print("******************************************************************")
        print("Starting feature selection. Initial number of features: " + str(self.initial_amount_of_features))

        if(len(subset) > 0):
            self.predefined_subset_selection(subset)


        flag_vis = False
        self.feature_visualization(flag_vis)

        if self.load_flag:

            self.load_features()

        self.get_filtered_data_with_ID()

        self.pairwise_selection()

        print("Finished feature selection. Number of features after filtering is: " + str(len(self.filteredFeatures)))
        print("Remaining features names: ")
        print(self.filteredFeatures)
        print("******************************************************************")

        self.get_selected_features_categories()

        return


    def load_features(self):

        self.filteredFeatures = list(self.loaded_features)

        return self

    def get_selected_features_categories(self):

        gait_quant = set(self.filteredFeatures).intersection(set(__GAIT_QUANTITY_FEATURES__))
        print("gait_quant:")
        print(gait_quant)

        gait_quality = set(self.filteredFeatures).intersection(set(__GAIT_QUALITY_FEATURES__))
        print("gait_quality:")
        print(gait_quality)

        transitions = set(self.filteredFeatures).intersection(set(__TRANSITIONS_FEATURES__))
        print("transitions:")
        print(transitions)

        sleep = set(self.filteredFeatures).intersection(set(__SLEEP_FEATURE_NAMES_LIST__))
        print("sleep:")
        print(sleep)

        fragment = set(self.filteredFeatures).intersection(set(__FRAGMENTATION_FEATURE_NAMES__))
        print("fragment:")
        print(fragment)


        print("Selected Features by category:")
        print("Gait quality: " + str(len(gait_quality)) + " features")
        print("Gait quantity: " + str(len(gait_quant)) + " features")
        print("Transitions: " + str(len(transitions)) + " features")
        print("Sleep features: " + str(len(sleep)) + " features")
        print("Fragmentations: " + str(len(fragment)) + " features")
        print("******************************************************************")

    def correlation_selected_features(self,selected_features, data, labels):

        data = pd.read_csv(data)
        labels = pd.read_csv(labels)
        ## Filter PD
        labels = labels[labels["Subject_Visit"] == 1]
        # labels = labels[labels["IsPD"] == 1]

        labels["FileName"] = labels["subject_ID"]
        feature_list = list(selected_features)
        # feature_list = feature_list[1:10]
        feature_list.append("FileName")
        data_all = pd.merge(data, labels, on="FileName", how="inner")
        feature_list.append('UPDRSII')
        feature_list.append('UPDRSIII')
        feature_list.append('UPDRSTotal')
        feature_list.append('PIGD_score')
        feature_list.append('TD_score')

        data_filtered = data_all[feature_list]

        corr_matrix_all = data_all.corr()
        corr_matrix = data_filtered.corr()
        updrs_corr = corr_matrix["UPDRSIII"]


        fig, ax = plt.subplots(figsize=(13, 13))  # Sample figsize in inches
        # sns.heatmap(df1.iloc[:, 1:6:], annot=True, linewidths=.5, ax=ax)
        sns.heatmap(corr_matrix, annot=True, linewidths=.5, ax=ax)
        plt.show()
        plt.savefig()

    def get_filtered_data_with_ID(self):

        featureList = self.filteredFeatures # + ["SubjectWithCenterID"]
        filteredDataID = self.data[featureList]
        filteredDataID.to_csv(os.path.join(self.input_path, "debug_files", "filteredData.csv"))

        filteredDataLabelID = pd.merge(self.labels, self.data[featureList], left_index=True, right_index=True)
        filteredDataLabelID.to_excel(os.path.join(self.input_path, "debug_files", "trainFilteredDataLabels.xlsx"))

    def getFilteredFeatures(self, forceGMMFeatures = True):
        if(len(self.filteredFeatures) > 0):
            if forceGMMFeatures:
                self.filteredFeatures = self.filteredFeatures + __ENGINEERED_FEATURES__
            return self.data[self.filteredFeatures]
        else:
            raise ValueError

    def univariate_selection(self):

        print(" starting univariate feature selection:")
        self.remove_zero_variance_features()
        self.univariate_relation()
        print(" finished univariate feature selection. Number of features remaining: " + str(len(self.filteredFeatures)))
        return


    def remove_zero_variance_features(self):

        print("     Removing low variance features, using threshold of " + str(__LOW_VARIANCE_FEATURES_TRES__))
        toFilter = []

        for f in self.filteredFeatures:
            featureCol = self.data[f]
            colVar = np.var(featureCol)
            if(colVar <= __LOW_VARIANCE_FEATURES_TRES__):
                toFilter.append(f)

        self.filteredFeatures = list(set(self.filteredFeatures) - set(toFilter))

        print("     Finished removing low variance features. Number of features remaining: " + str(len(self.filteredFeatures)))


    def univariate_relation(self):

        print("     Performing univariate correlation feature selection...")

        # Set variable types
        # self.labels = self.labels.dropna().astype('float64')
        # self.data = self.data.astype('float64')

        # Variable type - 'float64', numpy object
        # feature_score = SelectKBest(score_func = __UNIVARIATE_TEST_DICT__[__MODELING_MODE__]).fit(self.data[self.filteredFeatures].to_numpy(), np.ravel(self.labels.to_numpy())).scores_

        feature_score = SelectKBest(score_func = __UNIVARIATE_TEST_DICT__[__MODELING_MODE__]).fit(self.data[self.filteredFeatures].astype('float64'), np.ravel(self.labels.astype('float64'))).scores_
        # z = select.fit_transform(self.data, self.labels)
        df_feature_score = pd.DataFrame(pd.Series(feature_score))

        feature_score_dict = {"FeatureNames":self.filteredFeatures, "FeatureScore":df_feature_score.values.transpose()[0]}
        self.featureScores = pd.DataFrame(feature_score_dict)

        # Uncomment to observe histogram
        # plt.hist(self.featureScores["FeatureScore"])
        # plt.show()

        # plt.plot(sorted(self.featureScores["FeatureScore"]))
        # plt.show()

        maximal_features_number = min(int(len(self.data) / 2), len(self.filteredFeatures)-1,int(np.around(len(self.filteredFeatures)*50/100)))

        features_score_thres = sorted(df_feature_score.values.transpose()[0], reverse=True)[maximal_features_number]

        self.featureScores = self.featureScores.sort_values(by = "FeatureScore",ascending=False)


        # self.featureScores = self.featureScores[self.featureScores["FeatureScore"] > features_score_thres]

        # self.filteredFeatures = list(self.featureScores["FeatureNames"])
        # select_k = int(np.round(np.sqrt(len(self.data)) / 2))

        select_k = self.k_features
        self.featureScores = self.featureScores.iloc[0:select_k]
        self.filteredFeatures = list(self.featureScores["FeatureNames"][0:select_k])
        
        print("     Finished univariate correlation feature selection. Number of features remaining: " + str(len(self.filteredFeatures)))


    # remove correlated features
    def pairwise_selection(self, threshold: int = 0.75): #ToDO Do we choose features that might be correlated but clinicaly significant?

        print(" Performing pairwise correlation feature selection...")

        corr_df = self.data[self.filteredFeatures].astype(float).corr(method='pearson')
        corr_df['2nd_large'] = corr_df.apply(lambda row: row.nlargest(2).values[-1], axis=1)
        mask_threshold: np.ndarray = np.where(abs(corr_df.values) > threshold, 1, 0)
        corr_df['depend#'] = mask_threshold.sum(axis=1) - 1
        df_corr_count = pd.DataFrame(np.unique(corr_df['depend#'].values), columns=['depend_level'])
        bincount = np.bincount(corr_df['depend#'].values)
        df_corr_count['count'] = bincount[df_corr_count['depend_level']]
        sorted_corr_df = corr_df.sort_values(by=['depend#', '2nd_large'], ascending=[True, True])
        independent_set = set()
        ls: list = []
        for depend_level in df_corr_count['depend_level']:
            row_feature_indx = sorted_corr_df[sorted_corr_df['depend#'] == depend_level].index
            if depend_level == 0:
                independent_set = independent_set.union(row_feature_indx)
                continue
            for row in row_feature_indx:
                # get the features indices that has correlation greater than threshold with the feature in row
                row_series = sorted_corr_df.loc[row].drop(['depend#', '2nd_large'])
                col_feature_indx = row_series[abs(row_series) > 0.75].index
                corr_set = set(col_feature_indx)
                if independent_set.isdisjoint(corr_set):
                    independent_set.add(row)
        ls.append([*independent_set, ])
        independent_indx_list = list(itertools.chain.from_iterable(ls))
        self.filteredFeatures = independent_indx_list
        print(" number of selected features after pairwise feature selection stage: " + str(len(self.filteredFeatures)))


    def forward_selection(self):

        if __MODELING_MODE__ == "classification" or __MODELING_MODE__ == "multiclass":
            self.forward_selection_classification()
        elif __MODELING_MODE__ == "regression":
            self.forward_selection_regression()

    def forward_selection_classification(self,seed = 0):

        ffa_features_list = []
        clf = LogisticRegression(penalty="l2", solver="lbfgs", class_weight="balanced", random_state=0)
        max_scores = []
        # cv = StratifiedKFold(n_splits=__FOLDS_NUM__, shuffle=True, random_state=seed)

        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.3, shuffle=True,
                                                            random_state=seed)

        for i in range(min(__NUMBER_OF_FEATURES_FORWARD_SELECTION__,len(self.filteredFeatures))):
            scores = []
            for f in range(len(self.filteredFeatures)):
                if self.filteredFeatures[f] in ffa_features_list:
                    scores.append(0)
                else:
                    clf.fit(X_train[ffa_features_list + [self.filteredFeatures[f]]], y_train)
                    pred = clf.predict(X_test[ffa_features_list + [self.filteredFeatures[f]]])

                    if __MODELING_MODE__ == "classification":
                        score = f1_score(y_test, pred)
                    if __MODELING_MODE__ == "multiclass":
                        score = f1_score(y_test, pred, average="weighted")

                    # score = cross_validate(clf, self.data[ffa_features_list + [self.filteredFeatures[f]]], self.labels, cv=cv, scoring="f1_weighted") # .mean()
                    scores.append(score)

            k = np.argmax(scores)
            max_scores.append(max(scores))
            ffa_features_list.append(self.filteredFeatures[k])

        features_with_scores_dict = dict(zip(ffa_features_list,max_scores))
        print(" features ranking after forward feature selection stage: " + str(features_with_scores_dict))
        print("Close the opened figure to keep running")

        with open(os.path.join(self.input_path, "debug_files", str(len(features_with_scores_dict)) + "_" + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
            pickle.dump(features_with_scores_dict, outputFile)

        plt.plot(max_scores)
        plt.title("Forward selection scores - f1 score (scores change as samples are added)" + self.model_names)
        plt.show()



    def forward_selection_regression(self,clf, seed = 0):

        ffa_features_list = []
        clf = LinearRegression() # For each model
        max_scores = []
        cv = StratifiedKFold(n_splits=__FOLDS_NUM__, shuffle=True, random_state=seed)

        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.3, shuffle=True,
                                                            random_state=seed) #ToDo - Feature Selection: test_size = 0.2

        for i in range(min(__NUMBER_OF_FEATURES_FORWARD_SELECTION__,len(self.filteredFeatures))):
            scores = []
            for f in range(len(self.filteredFeatures)):
                if self.filteredFeatures[f] in ffa_features_list:
                    scores.append(0)
                else:
                    clf.fit(X_train[ffa_features_list + [self.filteredFeatures[f]]], y_train)
                    pred = clf.predict(X_test[ffa_features_list + [self.filteredFeatures[f]]])
                    # feature = self.filteredFeatures[f]
                    # feature = self.remove_feature_outliers(feature)

                    # Uncomment for a specific score
                    score = pearsonr(y_test.astype('float64'), pred)
                    # score = mean_absolute_error(y_test, pred)

                    # score = cross_validate(clf, self.data[ffa_features_list + [self.filteredFeatures[f]]], self.labels, cv=cv, scoring="f1_weighted") # .mean()
                    scores.append(score[0])
            print(i)

            k = np.argmax(scores)
            max_scores.append(max(scores))
            ffa_features_list.append(self.filteredFeatures[k])

        features_with_scores_dict = dict(zip(ffa_features_list,max_scores))
        print(" features ranking after forward feature selection stage: " + str(features_with_scores_dict))
        print(" forward feature selection model " + str(self.current_model))

        print("Close the opened figure to keep running")

        with open(os.path.join(self.input_path, "debug_files", str(len(features_with_scores_dict)) + "_" + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
            pickle.dump(features_with_scores_dict, outputFile)

        plt.plot(max_scores)
        plt.title("Forward selection scores - pearson correlation (scores change as samples are added)")
        plt.show()
        plt.savefig(os.path.join(self.input_path, "debug_files","Forward_selection_plots", str(__FILTERS__) +"_"+__LABELING_METHOD__ + str(len(features_with_scores_dict)) +"_" + str(self.current_model) + "_" + __DATE__+'.png'))
        plt.close()

        self.filteredFeatures = ffa_features_list

    def embedded_selection(self):

        print(" Perform embedded feature selection using random forest...")

        # # select top features with RF
        # random.seed(1234)
        # rf = RandomForestClassifier(n_estimators=100, random_state=0)
        # rf.fit(self.data[self.filteredFeatures], self.labels)
        # print(rf.feature_importances_)

        # select top features with RF
        random.seed(1234)
        clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=0)
        s = SelectFromModel(clf)
        s.fit(self.data[self.filteredFeatures], self.labels)
        selected = list(compress(__ALL_FEATURE_NAMES_LIST_WITH_ENGINEERED__, s.get_support()))

        self.filteredFeatures = selected

        print(" number of selected features after embedded feature selection stage: " + str(len(self.filteredFeatures)))

    '''
    If you want to use this function:
    Change the call to select_features in Main.py to include a list of features categories you want to use for modeling.
    Possible categories are the keys of __FEATURE_TYPE_DICT__ (see constants.py).
    Example for call: select_features(["fragmentation", "transitions", "quantity"])
    Will filter the features not in one of those categories.
    '''
    def predefined_subset_selection(self, subsetsToInclude):

        featureSet = set()
        for sub in subsetsToInclude:
            featureSet = featureSet.union(set(__FEATURE_TYPE_DICT__[sub]))

        self.filteredFeatures = list(featureSet)

    def forward_feature_selection_iterate_models(self):

        clf = None
        # model_name = self.model_names[0]

        for model_name in self.model_names:

            if model_name == "RFR":
                clf = RandomForestRegressor()
                self.current_model = model_name
                self.forward_selection_regression(clf, seed = 0)

            elif model_name == "LINR":
                clf = LinearRegression()
                self.current_model = model_name
                self.forward_selection_regression(clf, seed=0)

                # clf = LinearRegression()
            elif model_name == "RDG":
                clf = Ridge(alpha=0.1)
                self.current_model = model_name
                self.forward_selection_regression(clf, seed=0)

            elif model_name == "LASSO": #ToDo Uncomment for LASSO
                clf = Lasso()
                self.current_model = model_name
                self.forward_selection_regression(clf, seed=0)

            elif model_name == "XGB":  # ToDo Uncomment for LASSO
                clf = XGBRegressor()
                self.current_model = model_name
                self.forward_selection_regression(clf, seed=0)

        return self

    def mutual_info_regression(self,k):

        print("Mutual info")
        self.filteredFeatures = list(set(self.filteredFeatures).intersection(__ALL_FEATURE_NAMES_LIST__))
        self.data= self.data[self.filteredFeatures]
        data = self.data

        for col in self.filteredFeatures:
            na_s = data[col].isna().sum()
            if na_s>0:
                print(col)

        mi_scores = mutual_info_regression(data, self.labels, discrete_features='auto', n_neighbors=10, copy=True,random_state=0)

        feature_score_dict = {"FeatureNames": self.filteredFeatures,"FeatureScore": mi_scores}

        self.featureScores = pd.DataFrame(feature_score_dict)
        self.featureScores = self.featureScores.sort_values("FeatureScore",ascending=False)
        # self.featureScores = self.featureScores[self.featureScores["FeatureScore"] > 0.07]
        self.featureScores =  self.featureScores.iloc[:k,:]
        self.filteredFeatures = list(self.featureScores["FeatureNames"])

        features_with_scores_dict = dict(zip(self.featureScores["FeatureNames"], self.featureScores["FeatureScore"]))
        print(" features ranking after mutual information regression selection stage: " + str(features_with_scores_dict))

        with open(os.path.join(self.input_path, "debug_files",str(len(features_with_scores_dict)) + "_" + __LABELING_METHOD__+__DATE__ + '.p'),'wb') as outputFile:
            pickle.dump(features_with_scores_dict, outputFile)

        return self

    def feature_visualization(self,flag):

        if flag == True:

            for feature in self.filteredFeatures:

                plt.scatter(self.data[feature],self.labels)
                plt.title(feature)
                plt.xlabel(feature)
                plt.ylabel( str(__LABELING_METHOD__))
                plt.savefig(os.path.join(r"Path", str(__FILTERS__) + str(__LABELING_METHOD__) + feature +"_" + __DATE__ + '.png'))
                plt.close()


    def remove_feature_outliers(self,flag_outliers,feature):

        Q1, Q3 = self.data[feature].quantile([.25, .75])
        IQR = Q3 - Q1
        minimum = Q1 - 1.5 * IQR
        maximum = Q3 + 1.5 * IQR
        feature_without_outliers = self.data[feature][(self.data[feature].between(minimum, maximum, inclusive=True))]

        return feature_without_outliers

    def remove_feature_outliers_cluster(self,flag_outliers,feature):

        feature = "rngML_30sec_STD"
        XX = np.transpose([self.data[feature],self.labels])
        clf = OneClassSVM(kernel = 'rbf',gamma='auto').fit(XX)
        prediction = clf.predict(XX)

        plt.figure()

        plt.scatter(self.data[feature][prediction==1],self.labels[prediction==1])
        plt.scatter(self.data[feature][prediction==-1],self.labels[prediction==-1])


        plt.scatter(self.data[feature],self.labels)
        plt.legend(prediction)
        print(prediction)
        self.data = self.data[prediction==1]


    def MRMR_features_selection(self,path):

        best_list = io.loadmat(os.path.join(path,__LABELING_METHOD__, __LABELING_METHOD__ + "k" + str(__NUMBER_OF_FEATURES_MRMR__) + '_bestlist' + '.mat'))
        best_list = best_list['best_list']
        best_list = [x[0][0] for x in best_list]
        self.filteredFeatures = best_list

        return self

        
