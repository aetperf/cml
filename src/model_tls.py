import os

import git
import pandas as pd
from sklearn.datasets import load_boston


import model_cfg


def get_dataset_file_path():

    repo = git.Repo(os.path.realpath(__file__), search_parent_directories=True)
    repo_dir = repo.working_tree_dir
    data_dir = os.path.join(repo_dir, model_cfg.DATA_DIR)
    dataset_path = os.path.join(data_dir, model_cfg.DATA_FILE_NAME)

    return dataset_path


def fetch_dateset():

    # load the dataset from scikit-learn into a dataframe
    d = load_boston()
    df = pd.DataFrame(data=d.data, columns=d.feature_names)
    df["target"] = d.target

    # save the dataset into a parquet file
    dataset_path = get_dataset_file_path()
    df.to_parquet(dataset_path, compression=None)
