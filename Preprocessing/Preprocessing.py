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
import os
import numpy as np
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer,KNNImputer
from constants import __ALL_FEATURE_NAMES_LIST__, __LABELING_METHOD__, __IMPUTE_METHOD__,__NUMBER_OF_FEATURES_MUTUAL_INFO__


import pickle

__OUTLIERS_FEATURES_PERCENTAGE_FOR_SAMPLE_EXCLUSION__ = 0.15
__NUMBER_OF_DAYS_FOR_MINIMAL_QUALITY__ = 4
__NUMBER_OF_30SEC_BOUTS_FOR_MINIMAL_QUALITY__ = 40

# from constants import __ALL_FEATURES_A_


class Preprocessor():


    def __init__(self, X, y, input_path, run_mode):

        self.data = X.sort_values(by="SubjectID", axis = 0)
        self.labels = y
        self.sparseFeatures = [] # features which have too many missing values
        self.input_path = input_path
        self.run_mode = run_mode

    def initial_preprocessing(self):

        # self.manual_remove_samples()

        self.samples_quality_test()
        
        print("NaN before impute and outlier removal" + str(self.data.isna().sum().sum()))

        self.feature_normalization()

        self.impute_missing_values()

        self.impute_missing_values_KNN()

        self.remove_outliers()

        self.create_aligned_labels_vector()

        return self #, self.samples_ID

    def samples_quality_test(self):

        self.verify_minimal_information_features_and_samples() # Check feature and sample missing values
        self.check_samples_quality() # Check feature and sample contain >3 recording days

    # Measures for minimal quality of data:
    # * minimum number of days in recording for daily activity features
    # * minimum number of bouts for gait quality features
    # *
    def check_samples_quality(self):

        self.data = self.data[self.data["ValidDays12HR"] >= __NUMBER_OF_DAYS_FOR_MINIMAL_QUALITY__]
        self.data = self.data[self.data["NumberofWalkingBouts_30sec"] >= __NUMBER_OF_30SEC_BOUTS_FOR_MINIMAL_QUALITY__]
        return

    # Keep only features with values in at least %p1 of the samples
    # Keep only samples with values in at least  %p2 of the features:
    def verify_minimal_information_features_and_samples(self, feature_min_fraction = 0.75, sample_min_fraction=0.75):

        # self.data = self.data[[__ALL_FEATURES_A_]]
        features_list = list(set(self.data.columns).intersection(__ALL_FEATURE_NAMES_LIST__))

        col_stats = self.data.count(axis=0)

        sample_stats = self.data.count("columns")

        col_stats = col_stats.where(col_stats > feature_min_fraction * len(self.data)).dropna()
        sample_stats = sample_stats[sample_stats > sample_min_fraction * len(self.data.columns)]

        self.data = self.data[col_stats.index.tolist()]

        self.data = self.data.loc[sample_stats.index.tolist()]

        return col_stats.index.to_list()


    # Remove samples with high percentage of features having values which are considered as outliers:
    def remove_outliers(self):

        print("Removing outliers: ")
        outlier_count = pd.Series(False, index=self.data.index)
        feature_list = set(self.data.columns).intersection(__ALL_FEATURE_NAMES_LIST__)
        for col in feature_list:

            if 'm' in self.data[col]:
                print(col)

            Q1, Q3 = self.data[col].astype(float).quantile([.25, .75])
            IQR = Q3 - Q1
            minimum = Q1 - 1.5 * IQR
            maximum = Q3 + 1.5 * IQR
            mask = ~(self.data[col].between(minimum, maximum, inclusive=True))
            outlier_count = mask.astype(int) + outlier_count


        num_features = len(self.data.columns)

        self.data["outliers"] = outlier_count
        self.data["FileName"] = self.data.index

        outliers = self.data[self.data["outliers"] > __OUTLIERS_FEATURES_PERCENTAGE_FOR_SAMPLE_EXCLUSION__ * num_features]["FileName"]
        if len(outliers):
            print("Outliers removed:")
            print(list(outliers.index))
            
            outliers = pd.DataFrame(outliers)
            outliers.to_csv(r"outliers"+ "_" + __LABELING_METHOD__+ "_" + str(self.run_mode)+ '.csv')

        self.data = self.data[self.data["outliers"] <= __OUTLIERS_FEATURES_PERCENTAGE_FOR_SAMPLE_EXCLUSION__ * num_features]

        DataLabel = pd.merge(self.labels, self.data, left_index=True, right_index=True)
        DataLabel.to_csv(os.path.join(self.input_path, "debug_files", "OutliersDataLabels.csv"))


    def impute_missing_values(self): #ToDo: Add imputation method by using a subset defined by age (subjset of HY by age)

        # imp = SimpleImputer(missing_values=np.nan, strategy='median')
        # self.data=self.data.fillna(self.data.median())
        data_index = set(self.data.index)
        labels_index = set(self.labels.index)
        diff = labels_index.difference(data_index)
        common_index = labels_index.intersection(data_index)
        # print(diff)
        print(len(common_index))

        self.data = self.data.loc[list(common_index)]
        self.labels = self.labels.loc[list(common_index)]
        # self.data = self.data[self.data.index.isin(list(self.labels.index))]
        # self.labels = self.labels[self.labels.index.isin(list(self.data.index))]

        group_object = self.data.groupby([__IMPUTE_METHOD__], sort=False, as_index=False)

        if "Unnamed: 0" in self.data.columns:

            if (group_object.count()["Unnamed: 0"] == 1).sum()>0:


                multi_idx = group_object.count()["Unnamed: 0"][group_object.count()["Unnamed: 0"]==1].index[0]

                age_bin = list(group_object.groups.keys())[multi_idx]

                new_bin = age_bin+5

                subject_index = group_object.groups[age_bin].values[0]

                self.data["age_bin"].loc[self.data["FileName"] == subject_index] = new_bin


        self.data = self.data.groupby([__IMPUTE_METHOD__], sort=False, as_index=False).apply(lambda x: x.fillna(x.median(skipna=True))) #ToDo

        if "FileName" in self.data.columns:
            self.data.set_index("FileName", inplace=True, drop=False)

        self.data = self.data.sort_index(axis=0).drop([__IMPUTE_METHOD__], axis=1)
        # self.data = self.data.drop([__IMPUTE_METHOD__], axis=1)
        # self.data = self.data.dropna()
        return


    def impute_missing_values_KNN(self):

        print("KNN Impute")

        data_index = set(self.data.index)
        labels_index = set(self.labels.index)
        diff = labels_index.difference(data_index)
        common_index = labels_index.intersection(data_index)
        # print(diff)
        print(len(common_index))

        self.data = self.data.loc[list(common_index)]
        self.labels = self.labels.loc[list(common_index)]

        data_indices = pd.Series(self.data.index)
        
        features_to_impute = list(set(self.data.columns).intersection(__ALL_FEATURE_NAMES_LIST__))

        # Uncomment to add Meta Features
        # meta_features = [x for x in self.data.columns if x not in features_list]
        # features_to_impute = [i for i in self.data.columns if i not in meta_features]
        #self.scaledData = pd.DataFrame(self.scaledData, columns = features_list + meta_features, index=self.data.index)


        KNNimpute = KNNImputer(missing_values=np.nan, n_neighbors=20, weights='uniform', metric='nan_euclidean',copy=True, add_indicator=False)
        # imputed_temp = pd.DataFrame(KNNimpute.fit_transform(self.data[features_to_impute]), columns=features_to_impute)
        imputed_temp = pd.DataFrame(KNNimpute.fit_transform(self.data[features_to_impute])) #TODO
        imputed_temp = imputed_temp.assign(FileName=data_indices.values)
        imputed_temp = imputed_temp.set_index('FileName')

        self.data[features_to_impute] = imputed_temp

        print("Saving data, number of NaN:" + str(self.data.isna().sum().sum()))
        filteredDataLabel = pd.merge(self.labels, self.data, left_index=True, right_index=True)
        filteredDataLabel.to_csv(os.path.join(self.input_path, "debug_files", "ImputedDataLabels.csv"))
        # self.data = sklearn.impute.KNNImputer(self.data, missing_values=nan, n_neighbors=5, weights='uniform', metric='nan_euclidean',copy=True, add_indicator=False, keep_empty_features=False)

        return self


    def feature_normalization(self, method = "normalize"):

        features_list = list(set(self.data.columns).intersection(__ALL_FEATURE_NAMES_LIST__))
        
        if "female" in self.data.columns:

            data = self.data[features_list]
            features_list.remove("male")
            features_list.remove("female")
            ct = ColumnTransformer([("StandardScaler", StandardScaler(), features_list)], remainder='passthrough')
            
            
            data = data.loc[:, ~data.columns.isin(['female', 'male'])]
            male_female = self.data.loc[:, self.data.columns.isin(['female', 'male'])]

            data_indices = pd.Series(self.data.index)

            data_scaled = ct.fit_transform(data)
            data_scaled = pd.DataFrame(columns=features_list,data = data_scaled)
            data_scaled = data_scaled.assign(FileName=data_indices.values)
            data_scaled = data_scaled.set_index('FileName')

            # self.data = pd.merge(male_female,data_scaled,left_on= )
            self.data = pd.concat([male_female, data_scaled], axis=1)
            self.scaledData = self.data

        else:

            ct = ColumnTransformer([("StandardScaler", StandardScaler(), features_list)], remainder='passthrough')
            self.scaledData = ct.fit_transform(self.data)
            meta_features = [x for x in self.data.columns if x not in features_list]
            self.scaledData = pd.DataFrame(self.scaledData, columns=features_list + meta_features, index=self.data.index)

        self.data = self.scaledData #TODO Scaled Data

        if self.run_mode == "save":
            with open(os.path.join(self.input_path, "debug_files", "scaleObject" +__LABELING_METHOD__+str(__NUMBER_OF_FEATURES_MUTUAL_INFO__)+ '.p'), 'wb') as outputFile:
                pickle.dump(ct, outputFile)


    # def create_aligned_labels_vector(self, labelingMethod = __LABELING_METHOD__): # Function version 1 - works with impute by age
    #     import time
    #
    #     # if not os.path.join(self.input_path, "debug_files",time.strftime("%Y%m%d-%H")):
    #     #     os.mkdir(os.path.join(self.input_path, "debug_files",time.strftime("%Y%m%d-%H")))
    #     self.labels = self.labels.dropna()  #Added 08112022
    #     labels_for_model = self.labels[self.labels.index.isin(list(self.data.index))]
    #     Data_for_model = self.data[self.data.index.isin(list(self.labels.index))]

    def create_aligned_labels_vector(self, labelingMethod=__LABELING_METHOD__):

        import time

        # if not os.path.join(self.input_path, "debug_files",time.strftime("%Y%m%d-%H")):
        #     os.mkdir(os.path.join(self.input_path, "debug_files",time.strftime("%Y%m%d-%H")))
        self.labels = self.labels.dropna()  # Added 08112022
        labels_for_model = self.labels[self.labels.index.isin(list(self.data.index))]
        Data_for_model = self.data[self.data.index.isin(list(self.labels.index))]

        # labels_for_model.to_csv(os.path.join(self.input_path, "debug_files",time.strftime("%Y%m%d-%H"), "labels.csv"))

        self.labels = labels_for_model[labelingMethod]
        self.data = Data_for_model
        self.labels = self.labels.sort_index()
        self.data = self.data.sort_index()  # amit050122
        # self.samples_ID = labels_for_model["ID"]


        data_preprocessed = pd.merge( self.data, self.labels,on=self.labels.index)
        data_preprocessed.to_csv(os.path.join(self.input_path, "debug_files", __LABELING_METHOD__+"data_preprocessed.csv"))

        return self.data, self.labels


    def remove_poor_features_by_external_test_set(self, X_external, feature_min_fraction = 0.75): #ToDo thresh=0.75

        if self.run_mode == "save":
            col_stats = X_external.count(axis=0)
            col_stats = col_stats.where(col_stats > feature_min_fraction * len(X_external)).dropna()

            self.data = self.data[col_stats.index.tolist()]


    def impute_No_Impute(self):

        print("No Impute")

        self.data = self.data.dropna()

        data_index = set(self.data.index)


        print("NaN after drop: " + str(self.data.isna().sum().sum()))
        print("N samples: " + str(len(self.data)))


        labels_index = set(self.labels.index)
        diff = labels_index.difference(data_index)
        common_index = labels_index.intersection(data_index)
        # print(diff)
        print(len(common_index))

        self.data = self.data.loc[list(common_index)]
        self.labels = self.labels.loc[list(common_index)]

        DataLabel = pd.merge(self.labels, self.data, left_index=True, right_index=True)
        DataLabel.to_csv(os.path.join(self.input_path, "debug_files", "NoImputeDataLabels.csv"))


        return self