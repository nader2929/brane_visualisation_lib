import visualisation
import os
import shutil
from pandas import util as putil
import pandas as pd
from functools import wraps
import sys


def with_data_folder(func):
    @wraps(func)
    def create_data_run_rm_data(*args, **kwargs):
        data_folder = "./data"
        os.makedirs(data_folder, exist_ok=True)

        func(*args, **kwargs)
        shutil.rmtree(data_folder)
    return create_data_run_rm_data

@with_data_folder
def test_histogram_plot():
    df = putil.testing.makeDataFrame()
    input_data_path = "./data/test_df.csv"
    df.to_csv(input_data_path)
    output_plot_path = "./data/hist_plot.png"
    output = visualisation.histogram_plot("B", input_data_path, data_path_dst=output_plot_path)
    assert output is not None
    assert os.path.exists(output_plot_path)

