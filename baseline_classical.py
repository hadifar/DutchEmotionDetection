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


def init_logger(args):
    time = datetime.datetime.now()
    output_dir = "./results/results_{}_{}_{}_{}_{}_exp_{}_{}".format(
        args.task,
        time.month,
        time.day,
        time.hour,
        time.minute,
        args.experiment,
        args.model_name.split('/')[0],
        # str(args.crf),
        # str(args.epochs),
        # str(args.prev_history),
        # str(args.post_history),
        # str(args.lr)
    )

    log_dir = output_dir + '/logs'
    wandb.init(project="dutch_emotion_detection_crf_v2", name=output_dir.split('/')[-1])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return output_dir, log_dir


# def compute_metrics(labels, preds, le):
#     # precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(labels, preds)
#     precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')
#     precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average='micro')
#     precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average='weighted')
#     precision_s, recall_s, f1_s, _ = precision_recall_fscore_support(labels, preds, average='samples')
#
#     return {
#         'precision_sampled': precision_s,
#         'recall_sampled': recall_s,
#         'f1_sampled': f1_s,
#
#         'precision_micro': precision_micro,
#         'recall_micro': recall_micro,
#         'f1_micro': f1_micro,
#
#         'precision_macro': precision_macro,
#         'recall_macro': recall_macro,
#         'f1_macro': f1_macro,
#
#         'precision_weighted': precision_w,
#         'recall_weighted': recall_w,
#         'f1_weighted': f1_w,
#
#         'accuracy': accuracy_score(labels, preds)
#     }


def do_train_and_eval(args):
    # initialize logger
    output_dir, log_dir = init_logger(args)
    x_train, y_train, x_valid, y_valid, le = build_data(args)

    svc = SVC(kernel='linear', C=1, probability=True)
    svc = svc if args.task != 'response' else OneVsRestClassifier(svc)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ('svc', svc)], verbose=True)
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict_proba(x_valid)

    if args.task != 'response':
        predictions = [int(np.argmax(x)) for x in predictions]
    else:
        predictions = (predictions > 0.5).astype(int)

    pred = PredOutput(label_ids=y_valid, predictions=predictions)
    scores = compute_metrics(pred, le)

    os.makedirs(output_dir, exist_ok=True)
    dump(pipeline, os.path.join(output_dir, 'svm_{}.joblib'.format(output_dir.split('/')[-1])))
    # scores = compute_metrics(pred, le)
    wandb.log(scores)
    wandb.finish()

    # save results on json file
    with open(output_dir + '/{}_eval_scores.txt'.format(output_dir.split('/')[-1]), 'w') as outfile:
        for item in scores.items():
            outfile.write("{}: {}".format(item[0], item[1]))
            outfile.write('\n')
            print("{}: {}".format(item[0], item[1]))


def main(args):
    do_train_and_eval(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--task', type=str, default='subjectivity',
                        choices=['emotion', 'subjectivity', 'cause', 'response', 'valence', 'arousal'])
    parser.add_argument('--experiment', type=str, default='bytime')

    parser.add_argument('--prev_history', type=int, default=0, help='0 means single tweet, 1 means prev turns')
    parser.add_argument('--post_history', type=int, default=0, help='0 means single tweet, 1 means post turns')

    parser.add_argument('--train_data', type=str, default='data/date_time/stratifiedComp_time_train.json')
    parser.add_argument('--valid_data', type=str, default='data/date_time/stratifiedComp_time_test.json')

    parser.add_argument('--model_name', type=str, default="svm",
                        help='two type of classifier: neural-based (e.g., bert) and statistical based (e.g., svm)')

    args = parser.parse_args()
    main(args)
