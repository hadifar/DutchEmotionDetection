import argparse
import datetime
import json
import os
import random
import torch

from functools import partial

import numpy as np
from transformers import AutoTokenizer, TrainingArguments, Trainer, \
    AutoConfig, AutoModelForSequenceClassification, WEIGHTS_NAME, set_seed

from data_helper import build_data
from model import RobertaCRF, Encoder
from utils import SPECIAL_TOKENS, refine_label_format, compute_metrics


# from utils import le


def set_seed(args):
    # REPRODUCIBILITY
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logger(args):
    time = datetime.datetime.now()
    output_dir = "./results/results_{}_{}_{}_{}_{}_exp_{}_{}_crf{}_ep{}_h{}{}_lr{}_seed{}".format(
        args.task,
        time.month,
        time.day,
        time.hour,
        time.minute,
        args.experiment,
        args.model_name.split('/')[0],
        str(args.crf),
        str(args.epochs),
        str(args.prev_history),
        str(args.post_history),
        str(args.lr),
        str(args.seed),
    )

    log_dir = output_dir + '/logs'
    os.environ["WANDB_PROJECT"] = "dutch_emotion_detection_crf_v4"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return output_dir, log_dir


def load_from_ckp(ckp_path, le):
    config = AutoConfig.from_pretrained(ckp_path, num_labels=le.classes_)
    encoder = Encoder(config)
    # model = Model(encoder, encoder.transformer.config)
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')
    checkpoints = torch.load(os.path.join(ckp_path, WEIGHTS_NAME), map_location=device)
    encoder.load_state_dict(checkpoints['encoder'], strict=False)
    return encoder


def do_train_and_eval(args):
    # initialize logger
    output_dir, log_dir = init_logger(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    orig_num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens(SPECIAL_TOKENS)  # doesn't add if they are already there

    train_dataset, valid_dataset, le = build_data(args, tokenizer)

    print('tokenization finished...')

    if args.debug == 1:
        config = AutoConfig.from_pretrained(args.model_name)
        if hasattr(config, 'hidden_size'):
            config.hidden_size = 16
        if hasattr(config, 'intermediate_size'):
            config.intermediate_size = 4 * 16
        if hasattr(config, 'emb_dim'):
            config.emb_dim = 64
        config.num_attention_heads = 2
        config.num_hidden_layers = 2
        config.num_labels = le.classes_.size
        encoder = AutoModelForSequenceClassification.from_config(config)
    else:
        encoder = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=le.classes_.size,
                                                                     cache_dir=args.cache_dir)

    # resize embedding
    encoder.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
    # create model
    model = RobertaCRF(encoder, args.crf)

    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        num_train_epochs=args.epochs,  # total number of training epochs
        learning_rate=args.lr,
        gradient_accumulation_steps=32,
        per_device_train_batch_size=1,  # batch size per device during training
        per_device_eval_batch_size=1,  # batch size for evaluation
        warmup_steps=200,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=log_dir,  # directory for storing logs
        load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        # logging_steps=1,  # log & save weights each logging_steps
        # save_steps=100,

        metric_for_best_model='eval_f1_micro',
        greater_is_better=True,
        logging_strategy='epoch',
        save_strategy='epoch',
        evaluation_strategy="epoch",  # evaluate each `logging_steps`,
        # report_to="wandb",
        # run_name="emotion_crf"
    )

    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=valid_dataset,  # evaluation dataset
        # compute_metrics=compute_metrics  # the callback that computes metrics of interest
        # compute_metrics=compute_metrics  # the callback that computes metrics of interest
        compute_metrics=partial(compute_metrics, le=le)
    )

    trainer.train()
    trainer.evaluate()

    ##################################
    # save results on json file
    with open(output_dir + '/{}_eval_scores.json'.format(output_dir.split('/')[-1]), 'w') as outfile:
        pred_outputs = trainer.predict(valid_dataset)
        eval_pred = pred_outputs.predictions

        save_data = []
        for datapoint, pred in zip(valid_dataset, eval_pred):
            if args.task == 'response':  # inverse trans
                pred = [refine_label_format(p) for p in pred]
                pred = [p for p in pred if len(p) > 0]
                pred_label = [le.inverse_transform(np.expand_dims(p, 0)) for p in pred]
            else:
                pred_label = le.inverse_transform(refine_label_format(pred)).tolist()

            item = [
                tokenizer.batch_decode(datapoint['input_ids']),
                pred_label
            ]
            save_data.append(item)
        json.dump(save_data, outfile)


def main(args):
    set_seed(args)
    do_train_and_eval(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', type=str, default='response',
                        choices=['emotion', 'subjectivity', 'cause', 'response', 'valence', 'arousal'])
    parser.add_argument('--experiment', type=str, default='bytime')
    parser.add_argument('--crf', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=3)

    parser.add_argument('--prev_history', type=int, default=1, help='0 means single tweet, 1 means prev turns')
    parser.add_argument('--post_history', type=int, default=1, help='0 means single tweet, 1 means post turns')

    parser.add_argument('--train_data', type=str, default='data/date_time/stratifiedComp_time_train.json')
    parser.add_argument('--valid_data', type=str, default='data/date_time/stratifiedComp_time_test.json')
    parser.add_argument('--cache_dir', type=str, default='cache/')
    # "xlm-roberta-base"
    # "pdelobelle/robbert-v2-dutch-base"
    # "GroNLP/bert-base-dutch-cased"
    # "model_hub/xlm-roberta-base-ft-CSTwitter"
    parser.add_argument('--model_name', type=str, default="hadifar/xlm_pretrain",
                        help='two type of classifier: neural-based (e.g., bert) and statistical based (e.g., svm)')

    args = parser.parse_args()
    main(args)
