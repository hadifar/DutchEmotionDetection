import json
from itertools import chain

import torch

from utils import FROM_USER, FROM_OPERATOR, EMO_CLUSTER, get_label_encoder, clean_svm
from utils import clean_lm


class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.ds_len = len(encodings)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v) for k, v in self.encodings[idx].items()}
        if self.labels is not None:
            # item["label"] = torch.tensor([self.labels[idx]])
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return self.ds_len


def multilabel2singlelabel(emotion_clusters):
    """ if we have multiple emotion clusters, reduce it to one"""
    if len(emotion_clusters) == 1:
        return emotion_clusters
    else:
        emotion_clusters = sorted(emotion_clusters, key={c: i for i, c in enumerate(EMO_CLUSTER)}.get)
        return [emotion_clusters[0]]


def process_emotion_examples(ex, le, prev_history, post_history):
    ex_tweets = []

    if not ex['is_subjective']:
        return ex_tweets

    user_pointers = [i for i, t in enumerate(ex['turns']) if t['is_made_by_customer']]

    if len(user_pointers) < 2:
        return ex_tweets

    for p in user_pointers:
        # for t in ex['turns'][p]['emotions']['emo_clusters']
        tmp_labels = multilabel2singlelabel(ex['turns'][p]['emotions']['emo_clusters'])
        tmp_labels = le.transform(tmp_labels)
        if prev_history == 0 and post_history == 0:
            turns = [ex['turns'][p]]
        else:
            if p - prev_history < 0:
                turns = ex['turns'][0:p + 1 + post_history]
            else:
                turns = ex['turns'][p - prev_history - 1:p + 1 + post_history]

        tmp_tweets = []
        for turn in turns:
            prefix = FROM_USER if turn['is_made_by_customer'] else FROM_OPERATOR
            prefix = prefix + " " + " ".join([clean_lm(t['text']) for t in turn['tweets']])
            tmp_tweets.append(prefix)
        tmp_tweets = " ".join(tmp_tweets)
        ex_tweets.append([tmp_tweets, tmp_labels])

    return ex_tweets


def process_response_examples(ex, le):
    ex_tweets = []
    if not ex['is_subjective']:
        return ex_tweets

    operator_pointers = [i for i, t in enumerate(ex['turns']) if not t['is_made_by_customer']]
    for p in operator_pointers:
        response_strats = ex['turns'][p]['response_strats']
        # tmp_labels = response_strats
        tmp_labels = ['Other'] if len(response_strats) == 1 and response_strats[0] == 'None' else response_strats
        tmp_label = le.transform([tmp_labels])  # label binarizer requires 1x more dim
        tmp_label = tmp_label.astype(float).squeeze().tolist()
        tmp_tweets = clean_lm(" ".join([t['text'] for t in ex['turns'][p]['tweets']]))
        tmp_tweets = FROM_OPERATOR + " " + tmp_tweets
        ex_tweets.append([tmp_tweets, tmp_label])
    return ex_tweets


def process_valence_examples(ex, le):
    ex_tweets = []
    if not ex['is_subjective']:
        return ex_tweets

    user_pointers = [i for i, t in enumerate(ex['turns']) if t['is_made_by_customer']]
    for p in user_pointers:
        tmp_label = [str(ex['turns'][p]['emotions']['valence'])]
        tmp_label = le.transform(tmp_label)
        tmp_tweets = " ".join([t['text'] for t in ex['turns'][p]['tweets']])
        tmp_tweets = FROM_USER + " " + clean_lm(tmp_tweets)
        ex_tweets.append([tmp_tweets, tmp_label])

    return ex_tweets


def process_arousal_examples(ex, le):
    ex_tweets = []
    if not ex['is_subjective']:
        return ex_tweets

    user_pointers = [i for i, t in enumerate(ex['turns']) if t['is_made_by_customer']]
    for p in user_pointers:
        tmp_label = [str(ex['turns'][p]['emotions']['arousal'])]
        tmp_label = le.transform(tmp_label)
        tmp_tweets = " ".join([t['text'] for t in ex['turns'][p]['tweets']])
        tmp_tweets = FROM_USER + " " + clean_lm(tmp_tweets)
        ex_tweets.append([tmp_tweets, tmp_label])

    return ex_tweets


# resp stra: operator turns
def process_cause_examples(ex, le):
    ex_tweets = []

    label = [ex['cause']['cause_label'] if ex.get('has_cause', False) else "Other"]
    label = le.transform(label)
    first_user_turn = [i for i, t in enumerate(ex['turns']) if t['is_made_by_customer']][0]  # --> first user turn
    tweets = " ".join([t['text'] for t in ex['turns'][first_user_turn]['tweets']])
    tweets = FROM_USER + ' ' + clean_lm(tweets)

    ex_tweets.append([tweets, label])
    return ex_tweets


def process_subjectivity_examples(ex, le):
    # user_pointers = [i for i, t in enumerate(ex['turns']) if t['is_made_by_customer']]
    ex_tweets = []
    label = ["True" if ex['is_subjective'] else "False"]
    label = le.transform(label)
    tweets = " ".join([" ".join([t['text'] for t in t['tweets']]) for t in ex['turns'] if t['is_made_by_customer']])
    tweets = FROM_USER + ' ' + clean_lm(tweets)
    ex_tweets.append([tweets, label])
    return ex_tweets


def extract_example(task, ex, le, prev_history, post_history):
    if task == 'emotion':
        return process_emotion_examples(ex, le, prev_history, post_history)
    elif task == 'subjectivity':
        return process_subjectivity_examples(ex, le)
    elif task == 'cause':
        return process_cause_examples(ex, le)
    elif task == 'response':
        return process_response_examples(ex, le)
    elif task == 'valence':
        return process_valence_examples(ex, le)
    elif task == 'arousal':
        return process_arousal_examples(ex, le)
    else:
        raise Exception("Task not found...")


def load_data_from_file(data_path):
    with open(data_path) as inpfile:
        dataset = json.load(inpfile)
    return dataset


def extract_data(dataset, le, args):
    all_data = []
    for indx, ex in enumerate(dataset):
        d = extract_example(args.task, ex, le, args.prev_history, args.post_history)
        if len(d) != 0:  # if list is not empty
            all_data.append(d)

    return all_data


def process_data(train_data, valid_data, args):
    # in debug mode we use a subset of data
    if args.debug:
        valid_data = valid_data[:64]
        train_data = train_data[:64]

    le = get_label_encoder(args.task)
    train_ds = extract_data(train_data, le, args)
    valid_ds = extract_data(valid_data, le, args)

    # random.shuffle(all_data)
    return train_ds, valid_ds, le


def tokenize_4_transformers(tokenizer, ds):
    # item[0] id, item[1] txt, item[2] label
    # encoded_data = tokenizer([item[1] for item in ds], )
    encoded_data = [tokenizer([d[0] for d in item], truncation=True, padding=True, max_length=512) for item in ds]
    labels = [[t[1] for t in conv] for conv in ds]

    return QueryDataset(encoded_data, labels)


def tokenize_4_svm(ds):
    flatten_ds = list(chain(*ds))
    x_ = [clean_svm(item[0]) for item in flatten_ds]
    y_ = [item[1] for item in flatten_ds]

    return x_, y_


def tokenize_4_mc(ds):
    flatten_ds = list(chain(*ds))
    x_ = [item[0] for item in flatten_ds]
    y_ = [item[1] for item in flatten_ds]
    return x_, y_


def build_data(args, tokenizer=None):
    valid_file = load_data_from_file(args.valid_data)
    train_file = load_data_from_file(args.train_data)

    train_data, valid_data, le = process_data(train_file, valid_file, args)

    if args.model_name == 'svm':
        x_train, y_train = tokenize_4_svm(train_data)
        x_valid, y_valid = tokenize_4_svm(valid_data)
        return x_train, y_train, x_valid, y_valid, le
    elif args.model_name == 'mcb':
        x_train, y_train = tokenize_4_mc(train_data)
        x_valid, y_valid = tokenize_4_mc(valid_data)
        return x_train, y_train, x_valid, y_valid, le
    else:
        valid_dataset = tokenize_4_transformers(tokenizer, valid_data)
        train_dataset = tokenize_4_transformers(tokenizer, train_data)
        return train_dataset, valid_dataset, le
