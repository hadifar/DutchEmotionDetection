import re
import ssl

import emoji
import nltk
# from spacy.symbols import ORTH
import numpy as np
import spacy
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

tokenizer = TweetTokenizer(reduce_len=True)
spacy_model = spacy.load("nl_core_news_md")
stop_words = set(stopwords.words("dutch"))

EMO_CLUSTER = ["Anger", "Annoyance", "Disappointment", "Nervousness", "Gratitude", "Relief", "Joy", "Desire", "Neutral"]

SUBJECTIVITY = ['True', "False"]

RESPONSE = [['Request\\_information',
             'Explanation',
             'Cheerfulness',
             'Empathy',
             'Gratitude',
             'Apology',
             'Other',
             'Help\\_offline'
             ]]

CAUSE = ['Environmental\\_and\\_consumer\\_health',
         'Employee\\_service',
         'Breakdowns',
         'Digital\\_design\\_inadequacies',
         'Product\\_information',
         'Product\\_quality',
         'Delays\\_and\\_cancellations',
         'Other'
         ]

VALENCE = [str(i) for i in range(1, 6)]

AROUSAL = [str(i) for i in range(1, 6)]

# AT_CLIENT = "<at_client>"
# AT_COMPANY = "<at_company>"
AT_USER = "<at_user>"
URL = "<http_url>"
FROM_USER = "<from_client>"
FROM_OPERATOR = "<from_operator>"

SPECIAL_TOKENS = {
    'additional_special_tokens': [FROM_USER, FROM_OPERATOR, AT_USER, URL]}

SPECIAL_VOCAB = {FROM_USER, FROM_OPERATOR, AT_USER, URL}
COMPANIES = {"@BASE_nl", "@Telenet", "@mobilevikingsBE", "@OrangeBENL", "@proximus", "@Scarlet", "@delijn", "@NMBS",
             "@BrusselsAirport", "@FlyingBrussels", "@TUIflyBelgium"}


# def get_problem_type(task):
#     if task == 'response':
#         return 'multi_label_classification'
#     else:
#         return 'single_label_classification'


def get_label_encoder(task):
    if task == 'emotion':
        return LabelEncoder().fit(EMO_CLUSTER)
    elif task == 'subjectivity':
        return LabelEncoder().fit(SUBJECTIVITY)
    elif task == 'cause':
        return LabelEncoder().fit(CAUSE)
    elif task == 'response':
        return MultiLabelBinarizer().fit(RESPONSE)
    elif task == 'valence':
        return LabelEncoder().fit(VALENCE)
    elif task == 'arousal':
        return LabelEncoder().fit(AROUSAL)
    else:
        raise Exception("Task not found...")


def preprocess_emoji(text):
    return emoji.demojize(text)


def lower(text):
    return str(text).lower()


def tokenize(text):
    return tokenizer.tokenize(text)


def remove_stopwords(text):
    return [w for w in text if w.lower() not in stop_words]


def lemmatizer(text):
    # update_exc(BASE_EXCEPTIONS, SPECIAL_VOCAB)
    text = spacy_model(text)
    text_lemmas = [w.lemma_ for w in text]
    return text_lemmas


def anonymize(text):
    # replace http with URL
    text = re.sub(r"http\S+", URL, text)
    while text.find(URL + ' ' + URL) != -1:
        text = text.replace(URL + ' ' + URL, URL)

    # replace third party names with AT_OTHER
    text = re.sub(r"(?<!\w)@[\w+]{1,15}\b", AT_USER, text)

    return text


def clean_svm(text, anonymous=True):
    text = preprocess_emoji(text)
    text = lower(text)
    if anonymous is True:
        text = anonymize(text)
    text = lemmatizer(text)
    text = remove_stopwords(text)
    return " ".join(text)


def clean_lm(text, anonymous=True):
    text = preprocess_emoji(text)
    if anonymous is True:
        text = anonymize(text)
    return text


def refine_label_format(labels):
    labels = labels.flatten()
    labels = np.delete(labels, np.where(labels == -100))
    return labels.astype(int)


class PredOutput:
    def __init__(self, label_ids, predictions):
        self.label_ids = np.array(label_ids)
        self.predictions = np.array(predictions)


def compute_metrics(pred, le=None):
    *_, ls = pred.label_ids.shape
    # if type(le).__name__ != 'MultiLabelBinarizer':
    #     labels = refine_label_format(pred.label_ids)
    #     preds = refine_label_format(pred.predictions)
    # else:
    labels = np.reshape(pred.label_ids[np.where(pred.label_ids != -100)], (-1, ls))
    preds = np.reshape(pred.predictions[np.where(pred.predictions != -100)], (-1, ls))

    # precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average='micro')
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    #
    cls_report = classification_report(labels, preds, labels=le.transform(le.classes_),
                                       target_names=le.classes_.tolist())
    print(cls_report)
    #     precision_s, recall_s, f1_s = 0., 0., 0.
    # else:
    #     precision_s, recall_s, f1_s, _ = precision_recall_fscore_support(labels, preds, average='samples')

    return {
        # 'precision_sampled': precision_s,
        # 'recall_sampled': recall_s,
        'classification_report': cls_report,

        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,

        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,

        'precision_weighted': precision_w,
        'recall_weighted': recall_w,
        'f1_weighted': f1_w,

        'accuracy': accuracy_score(labels, preds)
    }
