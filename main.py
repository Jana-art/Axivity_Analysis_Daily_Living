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
import sys
import os
import argparse
import time as tm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from matplotlib import pyplot as plt
from Models.ModelingClassification import ModelsClassification
from Models.ModelingMulticlass import ModelsMulticlass
from Models.ModelingRegression import ModelsRegression
from Preprocessing.Preprocessing import Preprocessor
from FeatureSelection.Features import FeaturesSelector
import Misc
import messages
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

import pickle
import numpy as np
from constants import __LABELING_METHOD__, __DAY_NIGHT_FEATURES__, __TIMEFRAME_FILENAME__, __DAY_NIGHT_FILENAME__, __HY_BINS__, __HY_CLASS_NAMES, __ALL_FEATURE_NAMES_LIST__,\
    __IMPUTE_METHOD__, __FILTERS__, __MODELING_MODE__, __SCORING_METRICS__, __TEST_SCORES_ITERATIONS__, __CHECK_INPUT__, __DEMOGRAPHIC_FEATURES__,__MODELS__,__DATE__,__NUMBER_OF_FEATURES_MUTUAL_INFO__,__SEED__,__LOAD_TRAIN_FLAG__,\
    __LOAD_FEATURES_FLAG__, __DATE__

import warnings
warnings.filterwarnings("ignore")

def read_args():

    input_path = sys.argv[2]
    data_file = sys.argv[3]

    label_file = sys.argv[4]

    filter_conditions = sys.argv[5:]
    if type(filter_conditions) == type(str):
        filter_conditions = [filter_conditions]

    return input_path, data_file, label_file, filter_conditions

def check_labels_type(y):

    if __CHECK_INPUT__ == False:
        return
    if len(y) < 30:
        messages.write_message_too_little_data()
    if __MODELING_MODE__ == "classification":
        if sorted(list(pd.unique(y[__LABELING_METHOD__]))) != [0,1]:
            messages.write_message_labeling_problem()
    if __MODELING_MODE__ == "multiclass":
        val_list = pd.unique(y[__LABELING_METHOD__])
        if (not all([isinstance(s, (int, np.integer)) for s in val_list])) or (len(val_list) > np.max([len(y) / 30, 10])):
            messages.write_message_labeling_problem()
    if __MODELING_MODE__ == "regression":
        if len(pd.unique(y[__LABELING_METHOD__])) < 10:
            messages.write_message_labeling_problem()


def read_data(input_path, data_file, labels_file, labels_type = __LABELING_METHOD__):

    filter_conditions = __FILTERS__
    df_X = pd.read_csv(os.path.join(input_path, data_file))
    df_y = pd.read_csv(os.path.join(input_path, labels_file),encoding='latin1')

    #NEW GATE SPEED
    newFeatrues = pd.read_csv(os.path.join(r"Assaf GaitSpeed","ALL SPEED FEATURES.csv"))

    # apply filter on data and labels (e.g. only PD patients, only specific center):
    if filter_conditions != []:
        print("DATA FILTER APPLIED: " + str(filter_conditions))
        for f in filter_conditions:
            filter_name, filter_value = f.split("_")
            df_y = df_y[df_y[filter_name] == int(filter_value)]


    df_X.set_index("FileName", inplace=True, drop=False)

    #NEW Feature
    newFeatrues.set_index("SubjectID", inplace=True, drop=True)

    #df_y.set_index("subject_ID", inplace=True)
    df_y.set_index("subject_ID", inplace=True)

    # align data with labels (after possivle filtering):
    df_X = df_X[df_X.index.isin(list(df_y.index))]

    df_X = pd.merge(df_X, df_y[__DEMOGRAPHIC_FEATURES__], left_index=True, right_index=True, how='left')

    # add impute criteria information (will be removed later in preprocessing stage):
    if __IMPUTE_METHOD__ not in df_X.columns:

        df_X[__IMPUTE_METHOD__] = df_y[__IMPUTE_METHOD__]


    # NEW GATE Features
    df_X = pd.merge(df_X, newFeatrues, left_index=True, right_index=True, how='inner')

    # Keep only subject ID and the relevant labels:
    df_y = df_y[[labels_type]]

    execStrDescription = data_file.split('\\')[1].split('.')[0] + "_" + str(filter_conditions) + "_" + __LABELING_METHOD__

    return df_X, df_y, execStrDescription

def combine_features_dayNight_analysis(path, dType, timeFrameFileName, dayNightFileName):

    df_timeFrame = pd.read_csv(os.path.join(path, dType, timeFrameFileName))
    df_dayNight = pd.read_csv(os.path.join(path, dType, dayNightFileName))[["FileName"] + __DAY_NIGHT_FEATURES__]

    df_timeFrame[["BinaryIndex_Activity", "Active_Average_bout_duration_sec_Activity", "Sedentary_Average_bout_duration_sec_Activity", "Active_gini_index_Activity", "Sedentary_gini_index_Activity", "Active_Average_Hazard_Activity", "Sedentary_Average_Hazard_Activity", "KAR_transitions_from_active_to_rest_Activity", "KRA_transitions_from_rest_to_active_Activity"]]=[]
    frag_features_day = df_dayNight[["FileName","BinaryIndex_Activity_Day", "Active_Average_bout_duration_sec_Activity_Day", "Sedentary_Average_bout_duration_sec_Activity_Day", "Active_gini_index_Activity_Day", "Sedentary_gini_index_Activity_Day", "Active_Average_Hazard_Activity_Day", "Sedentary_Average_Hazard_Activity_Day", "KAR_transitions_from_active_to_rest_Activity_Day", "KRA_transitions_from_rest_to_active_Activity_Day"]]

    all_df = pd.merge(df_timeFrame, df_dayNight, on=["FileName"], how="inner")
    all_df_merged = pd.merge(all_df, frag_features_day, on=["FileName"], how="inner")

    all_df_merged.to_csv(os.path.join(path, dType, "allData.csv"))


    return "allData.csv"


def prepare_tables(path, dType):

    # read data file and prepare different files: all visits, only first visit, differences between visits
    if dType == "sleep":
        dataFile = os.listdir(os.path.join(path, dType))[0]
    else:
        dataFile = combine_features_dayNight_analysis(path, dType, __TIMEFRAME_FILENAME__, __DAY_NIGHT_FILENAME__)

    df = pd.read_csv(os.path.join(path, dType, dataFile))
    subjects = [int(x.split('_')[0].lstrip('0')) for x in df["FileName"]]
    visits = [int(x.split('_')[1].lstrip('0')) for x in df["FileName"]]
    df["SubjectID"] = subjects
    df["Visit"] = visits

    first_visit = df[df["Visit"] == 1].set_index("SubjectID", drop=False)
    second_visit = df[(df["Visit"] == 2) | (df["Visit"] == 99)].set_index("SubjectID", drop=False)
    third_visit = df[df["Visit"] == 3].set_index("SubjectID", drop=False)
    fourth_visit = df[df["Visit"] == 4].set_index("SubjectID", drop=False)
    fifth_visit = df[df["Visit"] == 5].set_index("SubjectID", drop=False)

    first_visit.sort_values(by="FileName", axis = 0, inplace=True)
    second_visit.sort_values(by="FileName", axis = 0, inplace=True)
    third_visit.sort_values(by="FileName", axis = 0, inplace=True)
    fourth_visit.sort_values(by="FileName", axis = 0, inplace=True)
    fifth_visit.sort_values(by="FileName", axis = 0, inplace=True)

    first_visit.to_csv(os.path.join(path, "mergedData", dType + "_data_first_visit.csv"))
    second_visit.to_csv(os.path.join(path, "mergedData", dType + "_data_second_visit.csv"))
    third_visit.to_csv(os.path.join(path, "mergedData", dType + "_data_third_visit.csv"))
    fourth_visit.to_csv(os.path.join(path, "mergedData", dType + "_data_fourth_visit.csv"))
    fifth_visit.to_csv(os.path.join(path, "mergedData", dType + "_data_fifth_visit.csv"))


def merge_features_table(path):

    fileType = ["first", "second", "third", "fourth", "fifth"]
    for fType in fileType:
        sleepFile=pd.read_csv(os.path.join(path, "mergedData", "sleep_data_" + fType + "_visit.csv"))
        axivityFile = pd.read_csv(os.path.join(path, "mergedData", "axivity_data_" + fType + "_visit.csv"))

        all_df = pd.merge(sleepFile, axivityFile, on=["FileName"], how="inner", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
        all_df.to_csv(os.path.join(path,"allData", "all_data_" + fType + "_visit.csv"))


def compute_labels(df_labels_data, labeling_method="isVR"):


    return


def get_file_id(subj, visit):

    visitDict = {1:"01", 2:"02", 3:"03", 4:"04", 5:"05", 9:"99"}

    subjectId = ''.join(map(str,np.zeros(4-len(str(subj))).astype(int))) + str(subj) + "_" + visitDict[visit]

    return subjectId

def split_to_bins(score):

    if score == 0:
        return 0
    elif score <= 5:
        return 1
    elif score <= 15:
        return 2
    elif score <= 30:
        return 3
    else:
        return 4

def create_age_column(df):

    df["age"] = np.floor((pd.to_datetime(df["visit date"]) - pd.to_datetime(df["BirthDate"])) / np.timedelta64(1, 'Y'))
    return df

def prepare_labels(path, fileName, demog):

    labels_path = os.path.join(path, "labels", fileName)
    df_labels_data = pd.read_csv(labels_path)
    df_labels_data["ID"] = df_labels_data.apply(lambda x: get_file_id(x.SubjectId, x.Visit), axis=1) #[["UPDRS1","UPDRS2","UPDRS3"]]

    df_demog = pd.read_csv(os.path.join(path, "labels", demog))
    df_demog["ID"] = df_demog.apply(lambda x: get_file_id(x.SubjectID, x.Visit), axis=1)

    df_labels_data = pd.merge(df_labels_data, df_demog, on=["ID"], how="left")

    df_labels_data = df_labels_data.dropna() #TODO:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    df_labels_data["isUnilateralStage"] = df_labels_data["HY"].apply(lambda x: 1<= x < 2)
    df_labels_data["UPDRS3Binning"] = df_labels_data["UPDRS 3"].apply(lambda x: split_to_bins(x))
    df_labels_data["HYBinning"] = df_labels_data["HY"].apply(lambda x: __HY_BINS__[x])
    df_labels_data["HYClasses"] = df_labels_data["HY"].apply(lambda x: __HY_CLASS_NAMES[x])
    df_labels_data = create_age_column(df_labels_data)

    df_labels_data = pd.concat([df_labels_data, pd.get_dummies(df_labels_data["HY"], prefix="HY", prefix_sep="-")], axis=1)
    df_labels_data = pd.concat([df_labels_data, pd.get_dummies(df_labels_data["HYBinning"], prefix="HYBinning", prefix_sep="-")], axis=1)
    df_labels_data = pd.concat([df_labels_data, pd.get_dummies(df_labels_data["UPDRS3Binning"], prefix="UPDRS3Binning", prefix_sep="-")], axis=1)

    df_labels_data.to_csv(os.path.join(path, "labels", "labelsScores.csv"))

    return

def create_folders(path):

    if not os.path.isdir(os.path.join(path, "data")):
        os.mkdir(os.path.join(path, "data"))

    if not os.path.isdir(os.path.join(path, "data", "allData")):
        os.mkdir(os.path.join(path, "data", "allData"))

    if not os.path.isdir(os.path.join(path, "data", "axivity")):
        os.mkdir(os.path.join(path, "data", "axivity"))

    if not os.path.isdir(os.path.join(path, "data", "labels")):
        os.mkdir(os.path.join(path, "data", "labels"))

    if not os.path.isdir(os.path.join(path, "data", "debug_files")):
        os.mkdir(os.path.join(path, "data", "debug_files"))

    if not os.path.isdir(os.path.join(path, "data", "mergedData")):
        os.mkdir(os.path.join(path, "data", "mergedData"))

    if not os.path.isdir(os.path.join(path, "data", "sleep")):
        os.mkdir(os.path.join(path, "data", "sleep"))

def get_execution_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("-config", "--configuration", help="running configuration type - initialize \ preparation \ analysis \ other")
    parser.add_argument("-path", "--inputpath", help="Path to data files")
    parser.add_argument("-data", "--data", help="Data file name")
    parser.add_argument("-labels", "--labels", help="Labels file name")
    parser.add_argument("-demographics", "--demographics", help="demographics file name")
    parser.add_argument("-runmode", "--runmode", help="save/load/other")

    args = parser.parse_args()

    print("configuration: {}\n inputpath: {}\n data: {}\n labels: {}\n demographics: {}\n".format(
        args.configuration,
        args.inputpath,
        args.data,
        args.labels,
        args.demographics,
        args.runmode
    ))

    return args

def main():

    args = get_execution_arguments()

    if args.configuration == "initialize":
        create_folders(args.inputpath)
        messages.write_message_initialize(args.inputpath)
        return

    # if args.configuration == "merge studies":


    if args.configuration == "preparation":
        prepare_tables(args.inputpath, "sleep")
        prepare_tables(args.inputpath, "axivity")
        merge_features_table(args.inputpath)
        # prepare_labels(args.inputpath, args.labels, args.demographics)
        return

    print("Starting pipeline with labeling by: " + __LABELING_METHOD__)

    if args.configuration == "analysis" or args.configuration == "preprocessing" or args.configuration == "feature_iteration" or args.configuration == "exhaustive_feature_selector" or args.configuration == "select_from_model":
        X, y, execDesc = read_data(args.inputpath, args.data, args.labels)
        check_labels_type(y)

        plot = False

        if plot == True:
          # deterministic random data
            _ = plt.hist(y[__LABELING_METHOD__], bins=30)  # arguments are passed to np.histogram
            plt.title("Histogram"+" "+ __LABELING_METHOD__+" "+X["Project"].values[0])
            plt.xlabel(__LABELING_METHOD__)
            plt.ylabel("N")
            plt.show()


        if args.runmode == "save":

            X = X.replace(r'^\s*$', np.nan, regex=True) # TODO replace empty spaces with NaN

            X = X.drop_duplicates("FileName")
            y = y.loc[y.index.drop_duplicates()]

            data_index = set(X.index)
            labels_index = set(y.index)
            diff = labels_index.difference(data_index)
            common_index = labels_index.intersection(data_index)
            print(diff)
            print(len(common_index))

            y = y.loc[list(common_index)]
            X = X.loc[list(common_index)]

            if __LOAD_TRAIN_FLAG__:

                X_train = pickle.load(open(os.path.join(args.inputpath, "debug_files", "X_train" + '.p'), "rb"))
                y_train = pickle.load(open(os.path.join(args.inputpath, "debug_files", "y_train" + '.p'), "rb"))
                X_test = pickle.load(open(os.path.join(args.inputpath, "debug_files", "X_test" + '.p'), "rb"))
                y_test = pickle.load(open(os.path.join(args.inputpath, "debug_files", "y_test" + '.p'), "rb"))
            else:

                Subjects = [x[0:-3] for x in X.index]
                Subject_groups = pd.Series(Subjects, index=X.index)
                gss = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=__SEED__)  # groupshuffle to seperate visits
                for train_idx, test_idx in gss.split(X, y, Subject_groups):
                    # print("TRAIN:", train_idx, "TEST:", test_idx)
                    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

                with open(os.path.join(args.inputpath, "debug_files", "X_train" + '.p'), 'wb') as outputFile:
                    pickle.dump(X_train, outputFile)
                with open(os.path.join(args.inputpath, "debug_files", "y_train" + '.p'), 'wb') as outputFile:
                    pickle.dump(y_train, outputFile)
                with open(os.path.join(args.inputpath, "debug_files", "X_test" + '.p'), 'wb') as outputFile:
                        pickle.dump(X_test, outputFile)
                with open(os.path.join(args.inputpath, "debug_files", "y_test" + '.p'), 'wb') as outputFile:
                        pickle.dump(y_test, outputFile)

                Processor = Preprocessor(X_train, y_train, args.inputpath, args.runmode)
                Processor.remove_poor_features_by_external_test_set(X_test)
                # X_processed, y_processed = Processor.initial_preprocessing()
                self = Processor.initial_preprocessing()
                #
                X = X_train
                y = y_train


            if __LOAD_FEATURES_FLAG__:

                # Updated features list
                loaded_features = pd.read_excel(os.path.join(args.inputpath, "debug_files",'FeaturePath',__LABELING_METHOD__ + ".xlsx"))

                loaded_features  = loaded_features["Feature"]

            if args.configuration == "feature_iteration":

                features_iterations_df = pd.DataFrame()

                X = X_train
                y = y_train

                for i in range(0, 100):

                    try:

                        print(i)


                        Subjects = [x[0:-3] for x in X.index]
                        Subject_groups = pd.Series(Subjects, index=X.index)
                        gss = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=i)  # groupshuffle to seperate visits
                        for train_idx_mi, test_idx_mi in gss.split(X, y, Subject_groups):
                            # print("TRAIN:", train_idx, "TEST:", test_idx)
                            X_train_MI, X_test_MI, y_train_MI, y_test_MI = X.iloc[train_idx_mi], X.iloc[test_idx_mi], y.iloc[train_idx_mi],y.iloc[test_idx_mi]

                        Processor = Preprocessor(X_train_MI, y_train_MI, args.inputpath, args.runmode)
                        # Processor.remove_poor_features_by_external_test_set(X_test)
                        # X_processed, y_processed = Processor.initial_preprocessing()

                        self = Processor.initial_preprocessing()

                        loaded_features = pd.read_excel(
                            os.path.join('FeaturePath', "selected_features_scores_histogram_by_cumscore" + __LABELING_METHOD__ + '.xlsx'))

                        loaded_features = loaded_features["Feature"]
                        __K_FEATURES__=40
                        features_score_thres = 80

                        # To select feature selection algorithm chose function in FeatureSelector
                        Selector = FeaturesSelector(self.data, self.labels, args.inputpath, __MODELS__[__MODELING_MODE__],loaded_features,features_score_thres,__K_FEATURES__)

                        Selector.select_features()


                    except:

                        print("Something went wrong at i = " + str(i))

                        continue

                    if i == 0:
                        features_iterations_df = Selector.featureScores.reset_index(drop=True)

                    elif i > 0:

                        features_iterations_df = pd.concat(
                            [features_iterations_df, Selector.featureScores.reset_index(drop=True)])

                features_iterations_df.to_csv(os.path.join(r"FeaturePath",__LABELING_METHOD__ + "_" + str("changing_seed" + "shuffle_True" +str(k)) + '.csv'))

            if args.configuration == "exhaustive_feature_selector":

                from mlxtend.feature_selection import ExhaustiveFeatureSelector
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.model_selection import GridSearchCV
                # from constants import __ALL_FEATURE_NAMES_LIST__,__LABELING_METHOD__


                Processor = Preprocessor(X_train, y_train, args.inputpath, args.runmode)
                # Processor.remove_poor_features_by_external_test_set(X_test)
                # X_processed, y_processed = Processor.initial_preprocessing()

                self = Processor.initial_preprocessing()

                X_train = self.data
                y_train = self.labels

                X = X_train[__ALL_FEATURE_NAMES_LIST__]

                efs = ExhaustiveFeatureSelector(RandomForestRegressor(n_estimators=350,max_features=0.333,max_depth=10, criterion='absolute_error'),scoring='r2',min_features=1,max_features=99, print_progress=True,cv=5)

                print('Starting EFS')

                efs.fit(X, y_train)

                print('Finished EFS. Saving Object.')

                with open(os.path.join(r"*PATH*_ \Data\data\debug_files","EFSObject" + __LABELING_METHOD__  + '.p'),'wb') as outputFile:
                    pickle.dump(efs, outputFile)
                # selected_features = X_train.columns[list(efs.best_idx_)]

                print('Best R2 score: %.2f' % efs.best_score_)
                print('Best subset (indices):', efs.best_idx_)
                print('Best subset (corresponding names):', efs.best_feature_names_)

            if args.configuration == "select_from_model":

                from sklearn.feature_selection import SelectFromModel
                from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
                from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, SGDRegressor, Lars, ElasticNet
                from xgboost import XGBRegressor

                # from constants import __ALL_FEATURE_NAMES_LIST__,__LABELING_METHOD__


                Processor = Preprocessor(X_train, y_train, args.inputpath, args.runmode)
                # Processor.remove_poor_features_by_external_test_set(X_test)
                # X_processed, y_processed = Processor.initial_preprocessing()

                self = Processor.initial_preprocessing()

                X_train = self.data
                y_train = self.labels

                ff = set(X.columns)
                ll = set(__ALL_FEATURE_NAMES_LIST__)

                features_all = ff.intersection(ll)

                X = X_train[features_all]  # Select from all features

                model = 'AdaBoostRegressor'
                estimator = AdaBoostRegressor(random_state=42, n_estimators=50)
                selector = SelectFromModel(estimator, max_features=40)
                selector = selector.fit(X, y_train)

                status = selector.get_support()

                print(model)
                print("Selected features:")
                print(X.columns[status])
                selector.transform(X)

                with open(os.path.join(r"*PATH*_ \Data\data\debug_files",model + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                    pickle.dump(list(X.columns[status]), outputFile)

                with open(os.path.join(
                        r"*PATH*_ \Data\data\debug_files",
                        model + 'Selector' + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                    pickle.dump(list(X.columns[status]), outputFile)

            ## RandomForestRegressor
                model = 'RandomForestRegressor'
                estimator = RandomForestRegressor(random_state=42, n_estimators=250)
                selector = SelectFromModel(estimator, max_features=40)
                selector = selector.fit(X, y_train)

                status = selector.get_support()

                print(model)
                print("Selected features:")
                print(X.columns[status])
                selector.transform(X)

                with open(os.path.join(r"*PATH*_ \Data\data\debug_files",model + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                    pickle.dump(list(X.columns[status]), outputFile)

                with open(os.path.join(
                        r"*PATH*_ \Data\data\debug_files",
                        model + 'Selector' + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                    pickle.dump(list(X.columns[status]), outputFile)

                ## Lasso
                model = 'Lasso'
                estimator = Lasso()
                selector = SelectFromModel(estimator, max_features=40)
                selector = selector.fit(X, y_train)

                status = selector.get_support()

                print(model)
                print("Selected features:")
                print(X.columns[status])
                selector.transform(X)

                with open(os.path.join(
                        r"*PATH*_ \Data\data\debug_files",
                        model + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                    pickle.dump(list(X.columns[status]), outputFile)

                with open(os.path.join(
                        r"*PATH*_ \Data\data\debug_files",
                        model + 'Selector' + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                    pickle.dump(list(X.columns[status]), outputFile)

                ## BayesianRidge
                model = 'BayesianRidge'
                estimator = BayesianRidge()
                selector = SelectFromModel(estimator, max_features=40)
                selector = selector.fit(X, y_train)

                status = selector.get_support()

                print(model)
                print("Selected features:")
                print(X.columns[status])
                selector.transform(X)

                with open(os.path.join(
                        r"*PATH*_ \Data\data\debug_files",
                        model + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                    pickle.dump(list(X.columns[status]), outputFile)

                with open(os.path.join(
                        r"*PATH*_ \Data\data\debug_files",
                        model + 'Selector'+__LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                    pickle.dump(list(X.columns[status]), outputFile)

                ## SGDRegressor
                model = 'SGDRegressor'
                estimator = SGDRegressor()
                selector = SelectFromModel(estimator, max_features=40)
                selector = selector.fit(X, y_train)

                status = selector.get_support()

                print(model)
                print("Selected features:")
                print(X.columns[status])
                selector.transform(X)

                with open(os.path.join(
                        r"*PATH*_ \Data\data\debug_files",
                        model + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                    pickle.dump(list(X.columns[status]), outputFile)

                with open(os.path.join(
                            r"*PATH*_ \Data\data\debug_files",
                            model + 'Selector' + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                        pickle.dump(list(X.columns[status]), outputFile)

                ## Lars
                model = 'Lars'
                estimator = Lars()
                selector = SelectFromModel(estimator, max_features=40)
                selector = selector.fit(X, y_train)

                status = selector.get_support()

                print(model)
                print("Selected features:")
                print(X.columns[status])
                selector.transform(X)

                with open(os.path.join(
                        r"*PATH*_ \Data\data\debug_files",
                        model + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                    pickle.dump(list(X.columns[status]), outputFile)

                with open(os.path.join(
                            r"*PATH*_ \Data\data\debug_files",
                            model + 'Selector' + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                        pickle.dump(list(X.columns[status]), outputFile)


                ## LinearRegression

                model = 'LinearRegression'
                estimator = LinearRegression()
                selector = SelectFromModel(estimator, max_features=40)
                selector = selector.fit(X, y_train)

                status = selector.get_support()

                print(model)
                print("Selected features:")
                print(X.columns[status])
                selector.transform(X)

                with open(os.path.join(r"*PATH*_ \Data\data\debug_files",model + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                        pickle.dump(list(X.columns[status]), outputFile)

                with open(os.path.join(
                            r"*PATH*_ \Data\data\debug_files",
                            model + 'Selector' + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                        pickle.dump(list(X.columns[status]), outputFile)

                ## Ridge

                model = 'Ridge'
                estimator = Ridge()
                selector = SelectFromModel(estimator, max_features=40)
                selector = selector.fit(X, y_train)

                status = selector.get_support()

                print(model)
                print("Selected features:")
                print(X.columns[status])
                selector.transform(X)

                with open(os.path.join(
                        r"*PATH*_ \Data\data\debug_files",
                        model + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                    pickle.dump(list(X.columns[status]), outputFile)

                with open(os.path.join(
                        r"*PATH*_ \Data\data\debug_files",
                        model + 'Selector' + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                    pickle.dump(list(X.columns[status]), outputFile)

                ## ElasticNet

                model = 'ElasticNet'
                estimator = ElasticNet()
                selector = SelectFromModel(estimator, max_features=40)
                selector = selector.fit(X, y_train)

                status = selector.get_support()

                print(model)
                print("Selected features:")
                print(X.columns[status])
                selector.transform(X)

                with open(os.path.join(
                        r"*PATH*_ \Data\data\debug_files",
                        model + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                    pickle.dump(list(X.columns[status]), outputFile)

                with open(os.path.join(
                        r"*PATH*_ \Data\data\debug_files",
                        model + 'Selector' + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                    pickle.dump(list(X.columns[status]), outputFile)

                ## XGBRegressor
                model = 'XGBRegressor'
                estimator = XGBRegressor()
                selector = SelectFromModel(estimator, max_features=40)
                selector = selector.fit(X, y_train)

                status = selector.get_support()

                print(model)
                print("Selected features:")
                print(X.columns[status])
                selector.transform(X)

                with open(os.path.join(
                        r"*PATH*_ \Data\data\debug_files",
                        model + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                    pickle.dump(list(X.columns[status]), outputFile)

                with open(os.path.join(
                        r"*PATH*_ \Data\data\debug_files",
                        model + 'Selector' + __LABELING_METHOD__ + '.p'), 'wb') as outputFile:
                    pickle.dump(list(X.columns[status]), outputFile)

                return print('Finished select from model feature selection')

            X = X_train
            y = y_train



        if args.runmode == "load":
            X = pickle.load(open(os.path.join(args.inputpath, "debug_files", "X_test" + '.p'), "rb"))
            y = pickle.load(open(os.path.join(args.inputpath, "debug_files", "y_test" + '.p'), 'rb'))
            X_test = X


        Processor = Preprocessor(X,y, args.inputpath, args.runmode)
        Processor.remove_poor_features_by_external_test_set(X_test)
        # X_processed, y_processed = Processor.initial_preprocessing()
        self = Processor.initial_preprocessing()

        if args.runmode == "load":

            selectedFeatures = pickle.load(open(os.path.join(args.inputpath, "debug_files", execDesc + "selectedFeatures" + '.p'), 'rb'))
            test_data = pd.merge(self.labels, self.data[selectedFeatures], left_index=True, right_index=True)
            test_data.to_excel(os.path.join(self.input_path, "debug_files", "testFilteredDataLabels.xlsx"))
            # y = pickle.load(open(os.path.join(args.inputpath, "debug_files", "y_test" + '.p'), 'rb'))

        if args.runmode == "load":
            self.data.to_excel(os.path.join(args.inputpath, "debug_files", 'X_test_' + __LABELING_METHOD__ + ".xlsx"))
            self.labels.to_excel(os.path.join(args.inputpath, "debug_files", 'y_test_' + __LABELING_METHOD__ + ".xlsx"))
        elif args.runmode == "save":
            self.data.to_excel(os.path.join(args.inputpath, "debug_files", 'X_train_' + __LABELING_METHOD__ + ".xlsx"))
            self.labels.to_excel(os.path.join(args.inputpath, "debug_files", 'y_train_' + __LABELING_METHOD__ + ".xlsx"))
        # self.data.to_csv(r'')



        # if configuration == "all" or configuration == "clusterModels": #TODO
    #     cls = Clustering(X_processed, y_processed)
    #     X_enriched = cls.find_clusters_with_GMM()

    if args.configuration == "analysis" and args.runmode == "load": #TODO:POC

        X_processed = self.data
        y_processed = self.labels

        X_processed = X_processed[X_processed.notnull()]
        y_processed = y_processed[y_processed.notnull()]

        X_index = set(X_processed.index)
        y_index = set(y_processed.index)

        common = X_index.intersection(y_index)
        # Remove Nan
        X_processed = X_processed.loc[common]
        y_processed = y_processed.loc[common]

        X_processed = X_processed.dropna()
        y_processed = y_processed.dropna()

        X_processed = X_processed[X_processed.notnull()]
        y_processed = y_processed[y_processed.notnull()]

        X_index = set(X_processed.index)
        y_index = set(y_processed.index)

        common = X_index.intersection(y_index)
        # Remove Nan
        X_processed = X_processed.loc[common]
        y_processed = y_processed.loc[common]


        # Model = pickle.load(open(os.path.join(args.inputpath, "debug_files", "ModelObject" + __LABELING_METHOD__ +str(1)+ '.p'), "rb"))
        # Model = pickle.load(open(os.path.join(argsF.inputpath, "debug_files", "best_estimator_XGB" + '.p'), "rb"))
        featureList = pickle.load(open(os.path.join(args.inputpath, "debug_files", execDesc + "selectedFeatures"+  '.p'), 'rb'))

        X = X_processed[featureList]

        # Iterate models

        corr = []
        MAE = []
        R_2 = []
        N_samples = []
        pval = []
        params = list()
        model_name = list()

        for i in range(0, __TEST_SCORES_ITERATIONS__):
            n = i
            if os.path.isfile(
                    os.path.join(args.inputpath, "debug_files", "ModelObject" + __LABELING_METHOD__ + str(n) + '.p')):

                Model = pickle.load(open(
                    os.path.join(args.inputpath, "debug_files", "ModelObject" + __LABELING_METHOD__ + str(n) + '.p'),
                    "rb"))

                # List intrinsic feature importance

                if hasattr(Model, 'feature_importances_'):

                    feature_importance_table = pd.DataFrame(data={'Feature':featureList,'Score':Model.feature_importances_})
                    feature_importance_table = feature_importance_table.sort_values(by='Score',ascending=False)
                    feature_importance_table.to_csv(os.path.join(args.inputpath, "debug_files", "Feature_importance" + __LABELING_METHOD__ + str(n) + '.csv'))

                pred = Model.predict(X) #Best estimator

                corr_val = pearsonr(pred, y_processed)
                MAE_val = mean_absolute_error(pred, y_processed)
                R_2_val = r2_score(y_processed, pred)
                N_samples_val = len(pred)

                corr.append(corr_val[0])
                pval.append(corr_val[1])
                MAE.append(MAE_val)
                R_2.append(R_2_val)
                N_samples.append(N_samples_val)
                model_name.append(Model.__class__.__name__)

                try:
                    params.append(Model.get_params())
                except:
                    print("An exception occurred:" + Model.__class__.__name__)
                    params.append({'None'})
                    continue;

                plot = True
                if plot:

                    #

                    plt.show()

                    fig, ax = plt.subplots(figsize=(6.4, 4.8))

                    # ax.text(0.1, 0.9, 'text', size=15, color='purple')

                    ax.scatter(y_processed, pred, zorder=1, alpha=0.6, color='#0000CC')

                    m, b = np.polyfit(y_processed, pred, 1)

                    ax.plot(y_processed, m * y_processed + b, 'k')

                    # plt.plot(y_test, y_test)
                    csfont = {'fontname': 'Times New Roman'}

                    # plt.plot(y_val, yhat, color='k')
                    if corr_val[1] < 0.001:
                        ax.text(0.6 * max(y_processed), 0.5 * max(pred),
                                'Pearson R = ' + str(np.around(corr_val[0], 2)) + '\n' + 'Pvalue < 0.001', fontsize=32,
                                bbox=dict(facecolor='None', alpha=0.5),**csfont)
                    else:
                        ax.text(0.6 * max(y_processed), 0.5 * max(pred),
                                'Pearson R = ' + str(np.around(corr_val[0], 2)) + '\n' + 'Pvalue = ' + str(
                                    np.around(corr_val[0], 2)), fontsize=32, bbox=dict(facecolor='None', alpha=0.5),**csfont)


                    if __LABELING_METHOD__ == "UPDRSI":
                        label_name = 'MDS-UPDRS I'
                    elif __LABELING_METHOD__ == "UPDRSII":
                        label_name = 'MDS-UPDRS II'
                    elif __LABELING_METHOD__ == "UPDRSIII":
                        label_name = 'MDS-UPDRS III'
                    elif __LABELING_METHOD__ == "PIGD_score":
                        label_name = 'PIGD score'
                    elif __LABELING_METHOD__ == "TD_score":
                        label_name = 'TD score'

                    plt.suptitle("Regression model, label: " + label_name, fontsize=40,**csfont)

                    # plt.title("Correlation: " + str(sl_corr),fontsize=8)
                    plt.xlabel("Actual " + label_name, fontsize=37,**csfont)
                    plt.ylabel("Predicted " + label_name, fontsize=37,**csfont)

                    plt.xticks(fontsize=35,**csfont)
                    plt.yticks(fontsize=35,**csfont)

                    # plt.rcParams["font.weight"] = "bold"
                    plt.rcParams['axes.linewidth']  = 1.2

                    fig.set_size_inches((12.4, 8.8), forward=False)

                    plt.savefig(
                        os.path.join(r"*PATH*_ \Results",
                                     "test_" + Model.__class__.__name__+str(n) + label_name + ".tif"),
                        dpi=600)

                    plt.close()

            else:

                continue;


        Pearson_statistic_total = [x for x in corr]
        Pearson_pval_total = [x for x in pval]

        MAE_total = [x for x in MAE]
        R_2_total = [x for x in R_2]
        N_samples_total = [x for x in N_samples]
        params_total = [x for x in params]
        model_name_total = [x for x in model_name]

        Mean_pearson = np.mean(Pearson_statistic_total)
        Mean_MAE = np.mean(MAE_total)
        Mean_R2 = np.mean(R_2_total)

        std_pearson = np.std(Pearson_statistic_total)
        std_MAE = np.std(MAE_total)
        std_R2 = np.std(R_2_total)

        result_dict = {'Model names': model_name_total, 'Model parameters': params_total,
                       'Pearson_statistic': Pearson_statistic_total,'P-value': Pearson_pval_total,
                       'MAE': MAE_total, 'R2': R_2_total, 'N_test': N_samples_total,
                       'Mean Pearson': Mean_pearson, 'Mean R2': Mean_R2, 'Mean MAE': Mean_MAE,
                       'std Pearson': std_pearson, 'std R2': std_R2, 'std MAE': std_MAE}

        for key in result_dict:
            N = len([result_dict[key]])
            if key == "Model names" or key == 'Model parameters':
                continue

            elif N == 1:
                result_dict[key] = np.round(result_dict[key], 2)

            else:  # rounding to K using round()
                i = 0
                N = len([result_dict[key]])
                for ll in range(0, N):
                    result_dict[key][ll] = np.round(result_dict[key][ll], 2)

        result_df = pd.DataFrame(data=result_dict)

        result_df.to_csv(
            os.path.join(r"*PATH*_ \Results",
                         "TestResults" + __LABELING_METHOD__ + __DATE__ + '.csv'))

        verbose = 1

        if verbose:
            print("test size: " + str(len(y_processed)))

            print("the average pearson correlation score of the model on the test set is: " + str(Mean_pearson))
            print("the average pearson MAE score of the model on the  test set is: " + str(Mean_MAE))
            print("the average R squared score of the model on the  test set is: " + str(Mean_R2))

        return

    if args.configuration == "analysis" or args.configuration == "FeatureSelection":


        X_processed = self.data
        y_processed = self.labels

        feature_thresh = 100
        __K_FEATURES__ = 40

        Selector = FeaturesSelector(self.data, self.labels, args.inputpath, __MODELS__[__MODELING_MODE__],loaded_features,feature_thresh,__K_FEATURES__)

        Selector.select_features()

        self.data = self.data[Selector.filteredFeatures] #TODO select features

        X_filtered = Selector.getFilteredFeatures(forceGMMFeatures=False)

        with open(os.path.join(args.inputpath, "debug_files", execDesc + "selectedFeatures" + '.p'), 'wb') as outputFile:
            pickle.dump(list(X_filtered.columns), outputFile)



    if args.configuration == "analysis" or args.configuration == "Modeling":

        messages.write_message_training_samples(len(X_filtered), len(X_filtered.columns), sum(y_processed.astype('float64')))
        if __MODELING_MODE__ == "classification":
            Model = ModelsClassification(X_filtered, y_processed)
        elif __MODELING_MODE__ == "multiclass":
            Model = ModelsMulticlass(X_filtered, y_processed)
        elif __MODELING_MODE__ == "regression":
            Model = ModelsRegression(X_filtered, y_processed)

        scores = []
        models = []

        for i in range(__TEST_SCORES_ITERATIONS__):

            start_time = tm.time()
            print("start time:"+ "--- %s seconds ---" % (start_time))

            model, chosen_model = Model.train_models(seed=i)
            models.append(model)
            # scores.append(model.best_score_)
            Model.get_CV_results(chosen_model,i)

            Model.MAE = {}
            Model.R_2 = {}
            Model.R_2_folds = {}
            Model.Pearson_corr = {}
            Model.Pearson_p = {}
            Model.N_train = {}
            Model.N_test = {}

            print("--- %s seconds ---" % (tm.time() - start_time))

        N_test = len(X_test)
        N_train = len(y_test)

        Model.get_all_results(chosen_model,N_train,N_test)


        print("--- %s seconds ---" % (tm.time() - start_time))


    if args.configuration == "misc":


        Misc.intersect_columns(args.data, args.labels)
        Misc.prepare_data()
        Misc.analyze_correlations(path)
        Misc.vis_features() #TODO


if __name__ == '__main__':
    main()