#!/usr/bin/env python3
from operator import rshift
from time import time
from unittest import result
import matplotlib.pyplot as plt
from matplotlib import rcParams
import csv
import numpy as np
import seaborn
import pandas
import pickle
import sys
import os
import ast
from parameters import Parameters
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import yaml

seaborn.set_style("whitegrid")
colours = seaborn.color_palette("pastel")
rcParams['figure.figsize'] = 15,10

def drop_columns(columns_to_drop, data_path : str = ""):
    global data
    # print(string_columns)
    if(data_path != ""):
        setup_data_with_path(data_path)
    if type(columns_to_drop) is not list:
        columns_to_drop = ast.literal_eval(f"{columns_to_drop}")

    data.drop(columns=columns_to_drop, inplace=True)

    data.to_csv(data_path.replace('.csv', '_red.csv'), index=False)

    return f"Results saved to file: {data_path.replace('.csv', '_red.csv')}, in data directory"


def analyse_column(column : str, printMessages : bool=False, data_path : str = "", data_path_dst:str=""):
    global data
    if(data_path != ""):
        setup_data_with_path(data_path)
    perc = []
    total_rows = len(data)
    counts = np.array(data[column].value_counts().values)
    values = np.array(data[column].value_counts().keys())
    for count in counts:
        p = round((count/total_rows)*100, 3)
        perc.append(p)

    testDf = pandas.DataFrame()
    testDf[column] = values
    testDf[f"{column}_counts"] = counts
    testDf[f"{column}_perc"] = perc
    dst_path = data_path_dst if data_path_dst else f"/data/{column}_analysed.csv"
    testDf.to_csv(dst_path, index=False)

    return f"Results saved to file: {dst_path}, in data directory"

def analyse_column_with_filter(column : str, filter_column : str, printMessages : bool=False, data_path : str = ""):
    global data
    if(data_path != ""):
        setup_data_with_path(data_path)
    totals = np.array(data[column].value_counts().values)
    values = np.array(data[column].value_counts().keys())
    filter_column_values = np.array(data[filter_column].value_counts().keys())

    total_true_per_column_cat = []
    total_false_per_column_cat = []
    perc_true_per_column_cat = []
    perc_false_per_column_cat = []
    perc_of_true_per_column_cat = []
    perc_of_false_per_column_cat = []
    
    total_true = data[filter_column].where(data[filter_column] == filter_column_values[0]).count()
    total_false = data[filter_column].where(data[filter_column] == filter_column_values[1]).count()
    for i in range(0, len(values)):
        no_true = data[column].where(data[column] == values[i]).where(data[filter_column] == filter_column_values[0]).count()
        no_false = data[column].where(data[column] == values[i]).where(data[filter_column] == filter_column_values[1]).count()
        total_true_per_column_cat.append(no_true)
        total_false_per_column_cat.append(no_false)

        p_true = round((no_true/totals[i])*100, 3)
        p_false = round((no_false/totals[i])*100, 3)
        perc_true_per_column_cat.append(p_true)
        perc_false_per_column_cat.append(p_false)

        p_of_true = round((no_true/total_true)*100, 3)
        p_of_false = round((no_false/total_false)*100, 3)
        perc_of_true_per_column_cat.append(p_of_true)
        perc_of_false_per_column_cat.append(p_of_false)

    testDf = pandas.DataFrame()
    testDf[column] = values
    testDf[f"total_{filter_column}_{filter_column_values[0]}"] = total_true_per_column_cat
    testDf[f"perc_{filter_column}_{filter_column_values[0]}"] = perc_true_per_column_cat
    testDf[f"perc_of_total_{filter_column}_{filter_column_values[0]}"] = perc_of_true_per_column_cat
    testDf[f"total_{filter_column}_{filter_column_values[1]}"] = total_false_per_column_cat
    testDf[f"perc_{filter_column}_{filter_column_values[1]}"] = perc_false_per_column_cat
    testDf[f"perc_of_total_{filter_column}_{filter_column_values[1]}"] = perc_of_false_per_column_cat
    testDf.to_csv(f"/data/{column}_analysed_with_filter.csv", index=False)

    return f"Results saved to file: /data/{column}_analysed_with_filter.csv, in data directory"

def replace_string_values(string_columns, data_path : str = ""):
    global data
    # print(string_columns)
    if(data_path != ""):
        setup_data_with_path(data_path)
    if type(string_columns) is not list:
        string_columns = ast.literal_eval(f"{string_columns}")
    for col in string_columns:
        values_set = set(data[col].values)        
        for i, v in enumerate(values_set):
            data[col] = data[col].replace(v, i)

    data.to_csv(f"/data/data_replaced_strings.csv", index=False)

    return "Results saved to file: /data/data_replaced_strings.csv, in data directory"

def train_and_test_classifier(target_column, test_size, model_type, exclude_first_column=True, replace_strings=True, data_path : str = "", data_path_dst:str=""):
    global data
    if(data_path != ""):
        setup_data_with_path(data_path)
    if replace_strings:
        replace_string_values(list(data.loc[:, data.dtypes == object].columns))
    target_data = data.pop(target_column)

    if exclude_first_column:
        data = data[data.columns[1:]]

    for col in data.columns:
        data[col] = data[col].fillna(data[col].median())

    training_data, test_data, training_result_data, test_result_data = model_selection.train_test_split(data, target_data, test_size=float(test_size), random_state=42)
    model_dict = {
        "dtree": DecisionTreeClassifier,
        "knn": KNeighborsClassifier,
        "rf": RandomForestClassifier,
        "mlp": MLPClassifier,
        "ada": AdaBoostClassifier
    }
    model = model_dict[model_type]()
    stringOutput = ""
    stringOutput += str(model.fit(training_data, training_result_data)) + "\n"
    stringOutput += f"Accuracy: {model.score(test_data, test_result_data)}\n"

    dst_path = data_path_dst if data_path_dst else f"/data/{model_type}_trained.pkl"

    with open(dst_path, "wb") as f:
        pickle.dump(model,f)

    stringOutput += f"Model saved to file: {dst_path}, in data directory. To use it do Y\n"
    return stringOutput

def load_classifier_and_predict(model_path : str, predict_data_path : str, prediction_column_name : str = "", replace_strings : bool = True, use_first_column_as_index : bool = True, predict_data_path_dst=""):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    global data
    data = pandas.read_csv(predict_data_path)

    if replace_strings:
        replace_string_values(list(data.loc[:, data.dtypes == object].columns))

    data.fillna(0, inplace=True)
    results = pandas.DataFrame()
    if use_first_column_as_index:
        results[data.columns[0]] = data[data.columns[0]]
        predictions =  model.predict(data.loc[:, data.columns != data.columns[0]])
    else:
        predictions =  model.predict(data)

    dst_path = predict_data_path_dst if predict_data_path_dst else "/data/model_predictions.csv"
    if prediction_column_name != "":
        results[prediction_column_name] = predictions
        results.to_csv(dst_path, index=False)
    else:
        results["predictions"] = predictions
        results.to_csv(dst_path, index=True, index_label="id")
    return f"Predictions saved to file: {dst_path}, in the data directory"


def setup_data_with_path(path):
    global data
    data = pandas.read_csv(path)
    return data

def setup_data(passed_data):
    global data
    data = passed_data

if __name__ == "__main__":
    command = sys.argv[1]

    parameters_dict = {
        "analyse_column": Parameters(analyse_column, 1, ["COLUMN"], 2, [("PRINTMESSAGES", False), ("DATA_PATH", "")]),
        "analyse_column_with_filter": Parameters(analyse_column_with_filter, 2, ["COLUMN", "FILTER_COLUMN"], 2, [("PRINTMESSAGES", False), ("DATA_PATH", "")]),
        "replace_string_values": Parameters(replace_string_values, 1, ["STRING_COLUMNS"], 1, [("DATA_PATH", "")]),
        "drop_columns": Parameters(drop_columns, 1, ["COLUMNS_TO_DROP"], 1, [("DATA_PATH", "")]),
        "train_and_test_classifier": Parameters(train_and_test_classifier, 3, ["TARGET_COLUMN", "TEST_SIZE", "MODEL_TYPE"], 3, [("EXCLUDE_FIRST_COLUMN", True), ("REPLACE_STRINGS", True), ("DATA_PATH", "")]),
        "load_classifier_and_predict": Parameters(load_classifier_and_predict, 3, ["MODEL_PATH", "PREDICT_DATA_PATH"], 3, [("PREDICTION_COLUMN_NAME", ""), ("REPLACE_STRINGS", True), ("USE_FIRST_COLUMN_AS_INDEX", True)])
    }
    # print(functions[command](argument))
    # print(f"Command: {command}")
    # print(f"Num Arguement: {parameters_dict[command].num_args}")
    # print(f"Arguements: {parameters_dict[command].arguments}")

    parameters_to_pass = []
    for i in parameters_dict[command].arguments:
        parameters_to_pass.append(os.environ[i])
    for i in parameters_dict[command].optional_arguments:
        if i[0] in os.environ:
            parameters_to_pass.append(os.environ[i[0]])
        else:
            parameters_to_pass.append(i[1])
    # setup_data_with_path(os.environ["data_path"])
    output = parameters_dict[command].function(*parameters_to_pass)
    print(yaml.dump({"resultOutput": output})) 
