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

from constants import __MODELING_MODE__, __LABELING_METHOD__, __TIMEFRAME_FILENAME__, __DAY_NIGHT_FILENAME__

def write_message_initialize(path):

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Directories were created in given path: " + path)
    print("Put the following files:")
    print("     " + __TIMEFRAME_FILENAME__)
    print("     " + __DAY_NIGHT_FILENAME__)
    print("in directory: " + path + "data\\axivity")
    print("And the file:")
    print("     SleepReport_PerWeek.csv")
    print("in directory: " + path + "data\\sleep")
    print("And put a labels file in:")
    print("in directory: " + path + "\\labels")
    print("Before running next steps")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

def write_message_training_samples(num_samples, num_features, positive_samples):

    print("**************************")
    print("Training models: ")

    print("Number of samples is: " + str(num_samples))
    print("Number of features is: " + str(num_features))
    if __MODELING_MODE__ == "classification":
        print("Number of positive samples: " + str(positive_samples))

def write_message_labeling_problem():

    print("***********************************************************************")
    print("PROBLEM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Mismatch between modeling mode: " + __MODELING_MODE__ + ", and labeling method: " + __LABELING_METHOD__)
    print("Change either of them in constants.py to be consistent!")
    print("(e.g. classification mode is for binary labels, regression mode is for numeric values, multiclass should be for few discrete classes ")
    print("If this error message is a mistake, change __CHECK_INPUT__ in constants.py to be False to avoid this input validity test")
    print("***********************************************************************")
    print("HASTA LA VISTA , BABY!")
    exit(1)

def write_message_score(method, score):

    print("*******************************\nTest results:")
    print(method + " is: " + str(score))

def write_message_too_little_data():
    print("***********************************************************************")
    print("Too little data - less than 30 samples!!!")
    print("If this error message is a mistake, change __CHECK_INPUT__ in constants.py to be False to avoid this input validity test")
    print("***********************************************************************")
    print("HASTA LA VISTA , BABY!")
    exit(1)

def write_message_no_model():
    print("**************************")
    print("Can't load model, no model learned!")
    print("**************************")


