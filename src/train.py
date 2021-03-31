import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold

import model_tls
import model_cfg


def run(model, metrics_file_name="metrics.txt"):

    # load the dataset from a parquet file
    dataset_path = model_tls.get_dataset_file_path()
    df = pd.read_parquet(dataset_path)
    feature_cols = [c for c in df.columns if c != "target"]

    # k-fold
    kf = KFold(n_splits=model_cfg.N_SPLITS, shuffle=True, random_state=model_cfg.RS)
    y_pred = np.zeros_like(df.target.values)
    for train_index, test_index in kf.split(df):
        train_df, eval_df = df.iloc[train_index], df.iloc[test_index]

        X_train, X_eval = train_df[feature_cols], eval_df[feature_cols]
        y_train = train_df["target"]

        model.fit(X_train, y_train)
        y_pred[test_index] = model.predict(X_eval)

    # display & write to text file
    mode = "w"
    for metric in [mean_absolute_error, mean_absolute_percentage_error]:
        s = f"{metric.__name__[:38]:>38} : \
            {metric(df.target.values, y_pred):15.8f}"
        print(s)
        with open(metrics_file_name, mode) as outfile:
            outfile.write(s + "\n")
        if mode == "w":
            mode = "a"


if __name__ == "__main__":

    # fetch the dataset if not found
    dataset_path = model_tls.get_dataset_file_path()
    if not os.path.isfile(dataset_path):
        model_tls.fetch_dateset()

    # instanciate the model
    model = RandomForestRegressor(random_state=model_cfg.RS, n_jobs=model_cfg.N_JOBS)

    # train and evaluate the model
    run(model)
