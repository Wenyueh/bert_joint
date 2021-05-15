import argparse
import collections
import gzip
import json
import os
from collections import OrderedDict
from urllib.parse import ParseResultBytes

from six import iteritems

# load the two files: gold file and prediction file
# gold: example_id: nqlabels
# pred: example_id: nqlabels
# check whether the ids are the same, if not, error
#
# compute long_answer_score for each id
# (1) check whether they are null or not
# if both non-null: check whether they are equal based on token offset
# return (1) gold_has_answer (2) pred_has_answer (3) is_correct (4) score
#
# compute short_answer_score for each id
# (1) check whether they are null or not
# since yes_no_answer are all "NONE"
# check whether they are equal spans by the same function
# return (1) gold_has_answer (2) pred_has_answer (3) is_correct (4) score
#
# based on compute_long_answer_score and compute_short_answer_score:
# compute recall, precision, f1
#


NQLabel = collections.namedtuple(
    "NQLabel",
    [
        "example_id",
        "long_answer_span",
        "short_answer_span_list",
        "yes_no_answer",
        "long_score",
        "short_score",
    ],
)


def safe_divide(x, y):
    if y == 0:
        return 0
    else:
        return x / y


def load_gold_labels(args):
    gold_dict = {}
    for each_file in os.listdir(args.eval_data_dir):
        with gzip.GzipFile(args.eval_data_dir + "/" + each_file, "rb") as f:
            for line in f:
                data = json.loads(line)

                example_id = data["example_id"]

                annotation_list = []

                for annotation in data["annotations"]:
                    long_answer = [
                        annotation["long_answer"]["start_token"],
                        annotation["long_answer"]["end_token"],
                    ]
                    short_answer_list = []
                    for short_answer in annotation["short_answers"]:
                        short_answer_list.append(
                            [short_answer["start_token"], short_answer["end_token"]]
                        )

                    gold_label = NQLabel(
                        example_id,
                        long_answer,
                        short_answer_list,
                        annotation["yes_no_answer"].lower(),
                        0,
                        0,
                    )

                    annotation_list.append(gold_label)

                gold_dict[example_id] = annotation_list

    return gold_dict


# select predictions
# compile to nqlable
def load_prediction_labels(args):
    nq_pred_dict = {}

    with open(args.eval_result_dir, "r") as f:
        predictions = json.load(f)

    # now for each instance I have one prediction
    # need to combine them so that each example_id has one prediction
    combined_predictions = {}
    best_predictions = {}
    for pred in predictions:
        if pred["example_id"] not in combined_predictions.keys():
            combined_predictions[pred["example_id"]] = [pred]
        else:
            combined_predictions[pred["example_id"]].append(pred)
    for id, preds in combined_predictions.items():
        best_pred = sorted(preds, key=lambda a: a["short_score"], reverse=True)[0]
        best_predictions[id] = best_pred

    for id, prediction in best_predictions.items():
        pred_item = NQLabel(
            id,
            prediction["long_answer_span"],
            [prediction["short_answer"]],
            "NONE",
            prediction["long_score"],
            prediction["short_score"],
        )
        nq_pred_dict[prediction["example_id"]] = pred_item

    return nq_pred_dict


def is_null_span(span):
    if span[0] == -1 and span[1] == -1:
        return True
    else:
        return False


# if the number of non-null long answers > 2
# then this example counts as having a long answer
def gold_has_long_answer(threshold, gold_list):
    number = 0

    for one_gold in gold_list:
        if not is_null_span(one_gold.long_answer_span):
            number += 1
    has_answer = gold_list and number >= threshold

    return has_answer


# if gold and pred has long answer
# then check whether the pred long answer matches any one of the gold long answer
def score_long_answer(args, gold_list, pred_list):
    # check if both have non-null/valid answers
    gold_has_answer = gold_has_long_answer(
        args.long_non_null_answer_threshold, gold_list
    )
    pred_has_answer = not is_null_span(pred_list.long_answer_span)

    score = pred_list.long_score

    is_correct = False

    if gold_has_answer and pred_has_answer:
        pred_long = pred_list.long_answer_span
        for gold_answer in gold_list:
            if pred_long == gold_answer.long_answer_span:
                is_correct = True
                break

    return gold_has_answer, pred_has_answer, is_correct, score


# if empty list, return true
# if no (-1,-1) in list, return true
def is_null_span_list(span_list):
    is_null = True
    for span in span_list:
        if not is_null_span(span):
            is_null = False
    if not span_list:
        is_null = True
    return is_null


def gold_has_short_answer(threshold, gold_list):
    number = 0

    for one_gold in gold_list:
        if (
            not is_null_span_list(one_gold.short_answer_span_list)
            or one_gold.yes_no_answer != "none"
        ):
            number += 1
    has_answer = gold_list and number >= threshold

    return has_answer


def score_short_answer(args, gold_list, pred_list):
    gold_has_answer = gold_has_short_answer(
        args.short_non_null_answer_threshold, gold_list
    )
    # in this model, only one short answer predict
    pred_has_answer = not is_null_span(pred_list.short_answer_span_list[0])

    is_correct = False

    score = pred_list.short_score

    if gold_has_answer and pred_has_answer:
        pred_short_answer = pred_list.short_answer_span_list
        for one_gold_list in gold_list:
            if one_gold_list.short_answer_span_list == pred_short_answer:
                is_correct = True
                break

    return gold_has_answer, pred_has_answer, is_correct, score


# first check whether the ids are the same
# score long answer
# score short answer
def score_predictions(args, gold_label, pred_label):
    gold_label_ids = gold_label.keys()
    pred_label_ids = pred_label.keys()

    assert sorted(gold_label_ids) == sorted(pred_label_ids)

    long_answer_stats = []
    short_answer_stats = []

    for id in gold_label_ids:
        gold = gold_label[id]
        pred = pred_label[id]

        long_answer_stats.append(score_long_answer(args, gold, pred))
        short_answer_stats.append(score_short_answer(args, gold, pred))

    # sort by score in order to faciliate finding score that maximizes R@P
    long_answer_stats = sorted(long_answer_stats, key=lambda x: x[-1], reverse=True)
    short_answer_stats = sorted(short_answer_stats, key=lambda x: x[-1], reverse=True)

    return long_answer_stats, short_answer_stats


# targets is a list of numbers in [0,1]
def compute_best_f1(stats, targets):

    total_has_gold = 0
    total_has_pred = 0
    total_correct = 0

    for has_gold, _, _, _ in stats:
        total_has_gold += has_gold

    max_recall = [0 for _ in targets]
    max_precision = [0 for _ in targets]
    max_score = [None for _ in targets]

    scores_to_stats = OrderedDict()
    for has_gold, has_pred, is_correct, score in stats:
        total_correct += is_correct
        total_has_pred += has_pred

        precision = safe_divide(total_correct, total_has_pred)
        recall = safe_divide(total_correct, total_has_gold)

        scores_to_stats[score] = (precision, recall)

    best_recall = 0
    best_precision = 0
    best_f1 = 0
    best_threshold = 0

    for score_threshold, (precision, recall) in iteritems(scores_to_stats):
        for i, target in enumerate(targets):
            if precision >= target and recall > max_recall[i]:
                max_recall[i] = recall
                max_precision[i] = precision
                max_score[i] = score_threshold

        f1 = 2 * safe_divide((recall * precision), recall + precision)
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_threshold = score_threshold

    return (
        (best_f1 * 100, best_precision * 100, best_recall * 100, best_threshold),
        list(zip(targets, max_recall, max_precision, max_score)),
    )


# compute precision, recall, f1 for long and short
def compute_plain_f1(long_answer_stats, short_answer_stats):
    # compute long
    total_long_gold = 0
    total_long_pred = 0
    total_long_correct = 0

    for gold_has_long, pred_has_long, is_correct, score in long_answer_stats:
        total_long_gold += gold_has_long
        total_long_pred += pred_has_long
        total_long_correct += is_correct

    long_precision = safe_divide(total_long_correct, total_long_pred)
    long_recall = safe_divide(total_long_correct, total_long_gold)
    long_f1 = 2 * safe_divide(
        (long_precision * long_recall), (long_precision + long_recall)
    )

    # compute short
    total_short_gold = 0
    total_short_pred = 0
    total_short_correct = 0

    for gold_has_short, pred_has_short, is_correct, score in short_answer_stats:
        total_short_gold += gold_has_short
        total_short_pred += pred_has_short
        total_short_correct += is_correct

    short_precision = safe_divide(total_short_correct, total_short_pred)
    short_recall = safe_divide(total_short_correct, total_short_gold)
    short_f1 = 2 * safe_divide(
        (short_precision * short_recall), (short_precision + short_recall)
    )

    return (
        long_precision * 100,
        long_recall * 100,
        long_f1 * 100,
        short_precision * 100,
        short_recall * 100,
        short_f1 * 100,
    )
