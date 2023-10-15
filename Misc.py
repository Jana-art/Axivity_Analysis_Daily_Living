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


import pickle
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr
import seaborn as sns

from collections import defaultdict

from constants import __FEATURE_TYPE_DICT__, __HY_BINS__, __HY_CLASS_NAMES, __MODELING_MODE__,__LABELING_METHOD__



def analyze_selected_features(self, features_path1, features_path2):

    with open(os.path.join(self.path, features_path1), "rb") as input_file1:
        features1 = pickle.load(input_file1)
    with open(os.path.join(self.path, features_path2), "rb") as input_file2:
        features2 = pickle.load(input_file2)

    print(features1)
    print(features2)

    print(set(features1).intersection(set(features2)))

def compare_specific_features(self, path, dataF, labelsF):

    features = ['slpAP_30sec_Prc10', 'rngML_30sec_Prc10', 'strRegAP_30sec', 'SparcAP_30sec_Prc10', 'rngML_30sec_Prc90', 'HRv_30sec_Prc90', 'WakeTimeNight', 'AniCVStrideTime_30sec', 'PercentWakeNight', 'strRegAP_30sec_Prc10', 'wdAP_30sec_Prc10', 'rngML_30sec', 'rmsML_30sec_Prc90', 'slpAP_30sec', 'rmsML_30sec', 'strRegAP_30sec_Prc90']
    df = pd.read_csv(os.path.join(path, dataF))
    labels = pd.read_csv(os.path.join(path, labelsF))

    df["ID"] = df["FileName"].apply(lambda x: x[0:6])
    df = df[features + ["ID"]]

    allDF = pd.merge(df, labels, on=["ID"], how="inner")

    pdDf = allDF[allDF["GROUP"] == 3]

    VR = pdDf[pdDf["binaryLabels"] == 1]
    TT = pdDf[pdDf["binaryLabels"] == 0]

    for f in features:
        print(f + " median for VR PD subjects is: " + str(np.median(VR[f])))
        print(f + " median for TT PD subjects is: " + str(np.median(TT[f])))

        plt.hist(VR[f], bins=10, alpha=0.5, label="VR")
        plt.hist(TT[f], bins=10, alpha=0.5, label="TT")
        plt.legend()
        plt.title(f)
        plt.show()


def analyze_correlations(path):

    data = pd.read_csv(path)
    # df = pd.read_csv(os.path.join(path, dataF))
    # labels = pd.read_csv(os.path.join(path, labelsF))
    #
    # df["ID"] = df["FileName"].apply(lambda x: x[0:6])
    # df_frag = df[__FEATURE_TYPE_DICT__["fragmentation"] + ["ID"]]
    #
    # df_frag.set_index("ID", inplace=True)
    # labels.set_index("ID", inplace=True)
    #
    # # labels = labels[labels.index.isin(list(df_frag.index))]
    #
    # merged = pd.merge(df_frag, labels, on="ID", how="inner")
    # merged = merged[merged["GROUP"] == 3].dropna()

    # for col in __FEATURE_TYPE_DICT__["fragmentation"]:
    #
    #     c1 = merged[col]
    #     c2 = merged["HOEHN"]
    #     corr = spearmanr(c1, c2)
    #
    #     print("pearson corr between: " + col + " & UPDRS3 is: " + str(corr))

    method = "spearman"
    corr_mat = data.corr(method)
    corr_mat_label = corr_mat[__LABELING_METHOD__]
    corr_mat_label = corr_mat_label.sort_values()
    corr_mat_label.to_csv(r"*PATH*_ \Data\data\debug_files\corr_mat_iter.csv")
    # corr_mat = corr_mat.loc[__FEATURE_TYPE_DICT__["fragmentation"]]
    # corr_mat = corr_mat[["UPDRS", "UPDRS3", "HOEHN"]]

    plt.figure(figsize=(15,6))
    heatmap = sns.heatmap(corr_mat, annot=True, cmap = "BrBG")
    heatmap.figure.subplots_adjust(left=0.3)
    heatmap.set_ylim(0,14)
    heatmap.set_title("Correlation: " + method + " Label: " + __LABELING_METHOD__, pad = 12)
    # plt.tight_layout()
    # plt.show()

    # plt.savefig() #, dpi=300, bbox_inches="tight")

    # merged.to_csv(os.path.join(path, "debug_files", "framentation_with_labels.csv"))


def check_overlaps(self, path1, path2):

    df_data = pd.read_csv(os.path.join(path2))[["FileName","StartTime","StopTime"]]
    df_session_times = pd.read_csv(os.path.join(path1))[["Subject","1st Session"]]
    count = 0

    for n in list(df_data["FileName"]):

        id = n.split("_")
        if id[0] != "01" or id[2] != "01":
            continue
        else:
            subject = "01_" + id[1]

            endTime = pd.to_datetime(df_data[df_data["FileName"] == n]["StopTime"], dayfirst=True)
            sessionStartTime = pd.to_datetime(df_session_times[df_session_times["Subject"] == subject]["1st Session"], dayfirst=True)
            if endTime.dt.date.values[0] >  sessionStartTime.dt.date.values[0]:
                print(subject) #, endTime.dt.date.values[0], sessionStartTime.dt.date.values[0], (endTime.dt.date.values[0] - sessionStartTime.dt.date.values[0]))
                count += 1
    print(count)


def get_falls_distribution(self, path1, path2, path3, path4):

    visits = pd.read_csv(path1)
    fall = pd.read_csv(path2)
    demo = pd.read_csv(path3)
    labels = pd.read_csv(path4)

    labels = labels[labels["lifeQualityResponsive"] == 1]["ID"].values

    vData = {}
    fData = {}
    bad_data = []

    for g in visits.groupby("ID"):
        vData[g[0]] = {}
        vData[g[0]][1] = pd.to_datetime(g[1][g[1]["visitid"] == 1]["SVSTDTC"])
        vData[g[0]][2] = pd.to_datetime(g[1][g[1]["visitid"] == 4]["SVSTDTC"])
        vData[g[0]][3] = pd.to_datetime(g[1][g[1]["visitid"] == 5]["SVSTDTC"])
        vData[g[0]][4] = pd.to_datetime(g[1][g[1]["visitid"] == 6]["SVSTDTC"])

        if len(vData[g[0]][4]) == 0:
            bad_data.append(g[0])

        vData[g[0]][5] = vData[g[0]][4] - pd.DateOffset(months=1)
        vData[g[0]][6] = vData[g[0]][4] - pd.DateOffset(months=2)
        vData[g[0]][7] = vData[g[0]][4] - pd.DateOffset(months=3)
        vData[g[0]][8] = vData[g[0]][4] - pd.DateOffset(months=4)


    for g in fall.groupby("ID"):

        if g[0] in bad_data:
            continue

        fData[g[0]] = defaultdict(int)

        for d in pd.to_datetime(g[1]["event_date"]):

            if len(vData[g[0]][2].values) > 0 and d < vData[g[0]][2].values[0]:
                fData[g[0]][1] += 1

            elif len(vData[g[0]][3].values) > 0 and d < vData[g[0]][3].values[0]:
                fData[g[0]][2] += 1

            elif len(vData[g[0]][8].values) > 0 and d < vData[g[0]][8].values[0]:
                fData[g[0]][7] += 1

            elif len(vData[g[0]][7].values) > 0 and d < vData[g[0]][7].values[0]:
                fData[g[0]][6] += 1

            elif len(vData[g[0]][6].values) > 0 and d < vData[g[0]][6].values[0]:
                fData[g[0]][5] += 1

            elif len(vData[g[0]][5].values) > 0 and d < vData[g[0]][5].values[0]:
                fData[g[0]][4] += 1

            elif len(vData[g[0]][4].values) > 0 and d < vData[g[0]][4].values[0]:
                fData[g[0]][3] += 1


    TT = demo[demo["arm"] == "TT"]["ID"].values
    VR = demo[demo["arm"] == "TT+VR"]["ID"].values

    # TT = list(set(TT).intersection(set(labels)))
    # VR = list(set(VR).intersection(set(labels)))

    first_per = []
    second_per = []
    third_per = []
    last_month_per = []
    two_month_per = []
    three_month_per = []
    four_month_per = []


    for s in TT:
        if s in fData.keys():
            first_per.append(fData[s][1])
            second_per.append(fData[s][2])
            last_month_per.append(fData[s][4])
            two_month_per.append(fData[s][5])
            three_month_per.append(fData[s][6])
            four_month_per.append(fData[s][7])
            third_per.append(fData[s][3])

    print(len(set(TT).intersection(set(fData.keys()))))
    print(np.mean(first_per))
    print(np.mean(second_per))
    print(np.mean(four_month_per))
    print(np.mean(three_month_per))
    print(np.mean(two_month_per))
    print(np.mean(last_month_per))
    print(np.mean(third_per))


    first_per = []
    second_per = []
    last_month_per = []
    third_per = []
    two_month_per = []
    three_month_per = []
    four_month_per = []

    for s in VR:
        if s in fData.keys():
            first_per.append(fData[s][1])
            second_per.append(fData[s][2])
            last_month_per.append(fData[s][4])
            two_month_per.append(fData[s][5])
            three_month_per.append(fData[s][6])
            four_month_per.append(fData[s][7])
            third_per.append(fData[s][3])

    print(len(set(VR).intersection(set(fData.keys()))))
    print(np.mean(first_per))
    print(np.mean(second_per))
    print(np.mean(four_month_per))
    print(np.mean(three_month_per))
    print(np.mean(two_month_per))
    print(np.mean(last_month_per))
    print(np.mean(third_per))


    print(fData)


def convert_subject_number(self, x):

    parts = x.split("_")
    new_subj_id = str(int(parts[0])*1000 + int(parts[1]))
    return new_subj_id

def convert_subject_id(self, x):

    parts = x.split("_")
    new_subj_id = str(int(parts[0])*1000)
    return new_subj_id

def process_subject_number(self, pathData):

    df = pd.read_csv(pathData)
    df["FileName"] = df["FileName"].apply(lambda x: self.convert_subject_number(x))
    df.to_csv(pathData)

def process_subject_ID(self, pathLabels):

    df = pd.read_csv(pathLabels)
    df["ID"] = df["ID"].apply(lambda x: self.convert_subject_number(x))
    df.to_csv(pathLabels)

def intersect_columns(self, path1, path2):

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    features = list(set(df1.columns).intersection(set(df2.columns)))
    df1 = df1[features]
    df2 = df2[features]

    df_merged = pd.concat([df1, df2], axis = 0, join="inner")

    df_merged.to_csv(path1 + "_merged.csv")

def intersect_columns_file_list(self, path):

    files = os.listdir(path)
    featuresList = []
    dfList = []
    dfFiltered = []

    for f in files:
        df = pd.read_csv(os.path.join(path, f))
        dfList.append(df)
        featuresList.append(set(df.columns))

    features = set.intersection(*featuresList)

    for i in range(len(dfList)):
        dfFiltered.append(dfList[i][features])

    df_merged = pd.concat(dfFiltered, axis = 0, join="inner")
    df_merged["FileName"] = df_merged["FileName"].apply(lambda x: self.convert_subject_number(x))
    df_merged["SubjectID"] = df_merged["FileName"]

    df_merged.to_csv(os.path.join(path, "canters_merged.csv"))


def get_subjectID(self, x):

    if "_" not in x:
        return x
    else:
        return int(x.split("_")[0])

def add_subj_ID_column(self, path):

    df = pd.read_csv(path)
    df["SubjectID"] = df["FileName"].apply(lambda x: self.get_subjectID(x))
    df.to_csv(path + "edit.csv")

def split_to_bins(self, score):

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

def prepare_labels(self, path):

    labels_path = os.path.join(path)
    df_labels_data = pd.read_csv(labels_path)
    HYClassNames = {0: 0, 1: 1, 1.5: 2, 2: 3, 2.5: 4, 3: 5, 4: 6, 5: 7}

    df_labels_data["isUnilateralStage"] = df_labels_data["HY"].apply(lambda x: 1 <= x < 2)
    df_labels_data["HYBinning"] = df_labels_data["HY"].apply(lambda x: __HY_BINS__[x])
    df_labels_data["HYClasses"] = df_labels_data["HY"].apply(lambda x: HYClassNames[x])
    df_labels_data["UPDRS3Binning"] = df_labels_data["UPDRS3"].apply(lambda x: self.split_to_bins(x))

    df_labels_data = pd.concat([df_labels_data, pd.get_dummies(df_labels_data["HY"], prefix="HY", prefix_sep="-")],
                               axis=1)
    df_labels_data = pd.concat(
        [df_labels_data, pd.get_dummies(df_labels_data["HYBinning"], prefix="HYBinning", prefix_sep="-")], axis=1)
    df_labels_data = pd.concat(
        [df_labels_data, pd.get_dummies(df_labels_data["UPDRS3Binning"], prefix="UPDRS3Binning", prefix_sep="-")],
        axis=1)
    df_labels_data.to_csv(path + "labelsScoresEdit.csv")


def store_pickles(path, objDict):

    for obj in objDict.keys():
        with open(os.path.join(path, "debug_files", obj + '.p'), 'wb') as outputFile:
            pickle.dump(objDict[obj], outputFile)

def load_debug_pickles(path, objList):

    objects = []
    for obj in objList:
        objects.append(pickle.load(open(os.path.join(path, "debug_files", obj + '.p')), 'rb'))
    return objects

def analyze_errors(path):

    if __MODELING_MODE__ == "classification":
        FP, FN, TP, TN = load_debug_pickles(path, ["FP_dict", "FN_dict", "TP_dict", "TN_dict"])

        alwaysWrongFP = []
        alwaysWrongFN = []
        moreFPthanTN = []
        moreFNthanTP = []

        for i in FP.keys():
            if i in TN.keys():
                if FP[i] >= TN[i]:
                    moreFPthanTN.append(i)
            else:
                moreFPthanTN.append(i)
                alwaysWrongFP.append(i)

        for j in FN.keys():
            if j in TP.keys():
                if FN[j] >= TP[j]:
                    moreFPthanTN.append(j)
            else:
                alwaysWrongFN.append(j)
                moreFPthanTN.append(j)

    # for col in discrete_demographic_features:



def correlation_selected_fetures(selected_fetures,data,labels):

    ## Filter PD

    labels = labels[labels["IsPD"] == 1]

    feature_list = list(selected_fetures["Feature"])

    # feature_list.append("FileName")
    feature_list.append("age")
    feature_list.append("male")
    feature_list.append("female")
    feature_list.append("UPDRSI")
    feature_list.append("UPDRSII")
    feature_list.append("UPDRSIII")

    # data_filtered = pd.merge(data[feature_list],labels,on="FileName",how="inner")
    data_filtered = data[feature_list]

    corr_matrix = data_filtered.corr()
    fig, ax = plt.subplots(figsize=(13, 13))  # Sample figsize in inches
    # sns.heatmap(df1.iloc[:, 1:6:], annot=True, linewidths=.5, ax=ax)
    sns.heatmap(corr_matrix, annot=True, linewidths=.5,ax=ax)
    plt.show()
    plt.savefig('r\heatmap.jpg')


