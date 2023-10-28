import argparse
import datetime
import os
import random

import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import wandb
from data_helper import build_data
from utils import compute_metrics, PredOutput

# from utils import le

# REPRODUCIBILITY
random.seed(0)
np.random.seed(0)


def main(args):
    _, y_train, _, y_valid, le = build_data(args)

    if args.task == 'response':
        _, values = np.where(np.array(y_train) == 1)  # row_index, col_index
        u, c = np.unique(values, return_counts=True)
        majority_label = u[c == c.max()]
        label = np.zeros([len(le.classes_)])
        label[majority_label] = 1.
        predictions = len(y_valid) * [label]
    else:
        values = np.array(y_train).flatten()
        u, c = np.unique(values, return_counts=True)
        majority_label = u[c == c.max()].tolist()
        predictions = len(y_valid) * [majority_label]

    # y_valid = np.expand_dims(np.array(y_valid), axis=-1)
    pred = PredOutput(label_ids=y_valid, predictions=predictions)
    scores = compute_metrics(pred)

    for item in scores.items():
        print("{}: {}".format(item[0], item[1]))


if __name__ == '__main__':
    # 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
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--task', type=str, default='subjectivity',
                        choices=['emotion', 'subjectivity', 'cause', 'response', 'valence', 'arousal'])
    parser.add_argument('--experiment', type=str, default='bytime')

    parser.add_argument('--prev_history', type=int, default=0, help='0 means single tweet, 1 means prev turns')
    parser.add_argument('--post_history', type=int, default=0, help='0 means single tweet, 1 means post turns')

    parser.add_argument('--train_data', type=str, default='data/date_time/stratifiedComp_time_train.json')
    parser.add_argument('--valid_data', type=str, default='data/date_time/stratifiedComp_time_test.json')

    parser.add_argument('--model_name', type=str, default="mcb",
                        help='majority class baseline')

    args = parser.parse_args()
    main(args)
