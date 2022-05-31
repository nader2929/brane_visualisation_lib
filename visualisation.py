#!/usr/bin/env python3
from time import time
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches
import csv
import numpy as np
from pyparsing import col
import seaborn
import pandas
import math
import sys
import os
import preprocessing
from parameters import Parameters
import yaml
import random

seaborn.set_style("whitegrid")
colours = seaborn.color_palette("pastel")
rcParams['figure.figsize'] = 15,10

def pie_plot_column(column : str, data_path : str):
    preprocessing.analyse_column(column, False, data_path)
    data = pandas.read_csv(f"/data/{column}_analysed.csv")
    rcParams['figure.figsize'] = 15,10
    titles = [f'{column} is {data[column][i]}: {data[f"{column}_perc"][i]}% ({data[f"{column}_counts"][i]})' for i in range(0, len(data))]

    titles_2 = [f'{data[f"{column}_perc"][i]}% ({data[f"{column}_counts"][i]})' for i in range(0, len(data))]

    plt.pie(data[f"{column}_perc"], colors=colours, labels=titles_2)
    plt.legend(titles, bbox_to_anchor=(1.5,1), loc="upper right")
    plt.savefig(f"/data/{column}_pie_plot.png", bbox_inches='tight')
    return f"Pie plot saved to file: /data/{column}_pie_plot.png, in data directory"
    #plt.show()

def pie_plot_column_with_filter(column : str, filter_column : str, data_path : str, legend=True):
    preprocessing.analyse_column_with_filter(column, filter_column, False, data_path)
    data = pandas.read_csv(f"/data/{column}_analysed_with_filter.csv")

    filter_column_values = []
    for i in range(1, len(data.columns), 3):
        filter_column_values.append(data.columns[i].split("_")[-1])

    rcParams['figure.figsize'] = 15,10
    for i in range(0, len(data)):
        plt.subplot(len(data), 1, i+1)  # row 1, column 2, count 1
        pie_data = [data[f"perc_{filter_column}_{k}"][i] for k in filter_column_values]
        labels = [f"{data[column][i]} {column} {filter_column} is {k} {data[f'perc_{filter_column}_{k}'][i]}% ({data[f'total_{filter_column}_{k}'][i]})" for k in filter_column_values]
        plt.pie(pie_data, labels=labels, colors=colours)
        total_for_col = sum([data[f'total_{filter_column}_{k}'][i] for k in filter_column_values])
        plt.title(f"{data[column][i]} ({total_for_col}) {column} {filter_column} rates")
    plt.savefig(f"/data/{column}_{filter_column}_rates.png", bbox_inches='tight')
    plt.clf()
    
    for k in filter_column_values:
        titles = [f"{column} {data[column][i]}: {data[f'perc_of_total_{filter_column}_{k}'][i]}% ({data[f'total_{filter_column}_{k}'][i]})" for i in range(0, len(data))]
        pie_data_perc_of = list(data[f'perc_of_total_{filter_column}_{k}'])
        plt.pie(pie_data_perc_of, colors=colours, labels=titles)

        if legend:
            titles_2 = [f"For {column} {data[column][i]} is {filter_column} {k}: {data[f'perc_of_total_{filter_column}_{k}'][i]}% ({data[f'total_{filter_column}_{k}'][i]})" for i in range(0, len(data))]
            plt.legend(titles_2, bbox_to_anchor=(1.5,1), loc="upper right")
        plt.title(f"{filter_column} is {k} distribution")
        plt.savefig(f"/data/{column}_{filter_column}_{k}_distribution.png", bbox_inches='tight')
        plt.clf()

    return f"Plots saved in data directory"

def scatter_plot_in_relation_to_column(column : str, data_path : str, group_by_column : str = " ",  alpha : float = 0.1):
    data = pandas.read_csv(data_path)
    counter = 1
    columns = list(data.columns)
    colours = ["b", "g", "r", "c", "m", "y", "darkred", "orangered", "crimson", "lightsteelblue"]
    patches = []
    for xcol in columns:
        i = 0
        if group_by_column != " ":
            for _,gr in data.groupby(group_by_column):
                seaborn.scatterplot(data=gr, x=xcol, y=column, alpha=float(alpha), color=colours[i])
                patches.append(mpatches.Patch(color=colours[i], label=f'{group_by_column}: {_}'))
                i += 1
                if i >= len(colours):
                    i = 0
        else:
            seaborn.scatterplot(data=data, x=xcol, y=column, alpha=float(alpha), color=colours[i])            

        plt.legend(handles=patches)
        patches = []
        plt.title(f"Plot:{counter} GROUPBY: {group_by_column}, X: {xcol}, Y: {column}")
        plt.savefig(f"/data/{column}_{xcol}_scatter.png", bbox_inches='tight')
        plt.clf()
    return f"Results saved to file: /data/{column}_scatter.png, in data directory"

def histogram_plot(column, data_path, hue_column=" ", data_path_dst=""):
    data = pandas.read_csv(data_path)
    if hue_column == " ":
        seaborn.histplot(data=data, x=column, legend=True)
    else:
        seaborn.histplot(data=data, x=column, hue=hue_column, legend=True, multiple="dodge")
    plt.title(f"Histogram of {column} values")
    dst_path = data_path_dst if data_path_dst else f"/data/{column}_histogram_plot.png"
    plt.savefig(dst_path, bbox_inches='tight')
    return f"Results saved to file: /data/{column}_histogram_plot.png, in data directory"
    #plt.show()

def line_plot(x_column, y_column, data_path):
    data = pandas.read_csv(data_path)
    seaborn.lineplot(data=data, x=x_column, y=y_column)
    plt.title(f"Line plot of X; {x_column} Y: {y_column}")
    plt.savefig(f"/data/x_{x_column}_y_{y_column}_line_plot.png", bbox_inches='tight')
    return f"Results saved to file: /data/{x_column}_line_plot.png, in data directory"
    #plt.show()

if __name__ == "__main__":
    command = sys.argv[1]

    parameters_dict = {
        "pie_plot_column": Parameters(pie_plot_column, 1, ["COLUMN"], 1, [("DATA_PATH", "")]),
        "pie_plot_column_with_filter": Parameters(pie_plot_column_with_filter, 2, ["COLUMN", "FILTER_COLUMN"], 2, [("DATA_PATH", ""), ("LEGEND", True)]),
        "scatter_plot_in_relation_to_column": Parameters(scatter_plot_in_relation_to_column, 2, ["COLUMN"], 3, [("DATA_PATH", ""), ("GROUP_BY_COLUMN", " "), ("ALPHA", 0.1)]),
        "histogram_plot": Parameters(histogram_plot, 1, ["COLUMN", "DATA_PATH"], 1, [("HUE_COLUMN", " ")]),
        "line_plot": Parameters(line_plot, 2, ["X_COLUMN", "Y_COLUMN"], 1, [("DATA_PATH", "")]),
    }
    # print(functions[command](argument))
    # print(f"Command: {command}")
    # print(f"Num Arguement: {parameters_dict[command].num_args}")
    # print(f"Arguements: {parameters_dict[command].arguments}")
    # print(f"Num Optional Arguement: {parameters_dict[command].num_optional_args}")
    # print(f"Optional Arguements: {parameters_dict[command].optional_arguments}")

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
# pie_plot_column("sex", "./data_for_training.csv")
# pie_plot_column_with_filter("sex", "survived", "./data_for_training.csv")
# scatter_plot_in_relation_to_column("age", "./data_for_training.csv", " ", 0.0)
# histogram_plot("sex", "./data_for_training.csv", "survived")
# line_plot("pid", "age", "./data_for_training.csv")