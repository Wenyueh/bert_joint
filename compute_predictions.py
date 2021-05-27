import gzip
import json
import os

import torch


# candidates_dict[example_id] = [long_answer_candidiates]
def compute_candidate_dict(eval_data_dir):
    candidate_dict = {}
    for each_file in os.listdir(eval_data_dir):
        with gzip.GzipFile(eval_data_dir + "/" + each_file, "rb") as f:
            for line in f:
                data = json.loads(line)
                candidate_dict[data["example_id"]] = data["long_answer_candidates"]
    return candidate_dict


# token_map_dict[example_id] = [token_map]
def compute_full_token_map_dict(eval_feature_dir):
    token_map_dict = {}
    with open(eval_feature_dir, "r") as f:
        data_points = json.load(f)
        if isinstance(data_points, dict):
            data_points = [data_points]
    for data in data_points:
        unique_index = tuple(data["unique_index"])  # list type is unhashable
        token_map = data["token_map"]
        token_map_dict[unique_index] = token_map
    return token_map_dict


def each_result(i, predictions, logits):
    one_prediction = {
        "start_predictions": torch.cat(
            (
                predictions["start_predictions"][0][i].unsqueeze(0),
                predictions["start_predictions"][1][i].unsqueeze(0) + 1,
            ),
            dim=0,
        ).tolist(),
        "end_predictions": torch.cat(
            (
                predictions["end_predictions"][0][i].unsqueeze(0),
                predictions["end_predictions"][1][i].unsqueeze(0) + 1,
            ),
            dim=0,
        ).tolist(),
        "type_predictions": predictions["type_predictions"][i].tolist(),
    }
    one_logit = {
        "start_logits": logits["start_logits"][i].tolist(),
        "end_logits": logits["end_logits"][i].tolist(),
        "type_logits": logits["type_logits"][i].tolist(),
    }

    return one_prediction, one_logit


# for each id
# find the best_n starts and best_n ends
# set max_answer_len
# if end < start, discard
# if token_map[end] or token_map[start] = -1, discard
# if end-start+1 > max_len, discard
# compute the score by adding the logits - cls logits
# both logits (batch * best_n_size) and predictions (batch * best_n_size)
# needs to be converted to (best_n_size)
# the short answer is the highest scored span
# for each long_answer_candidate in example"
# if the candidate is top_level and the short span is contained in it
# the candidate is the long span
# write a predicted_label dictionary containing:
# example_id
# long_answer start & end token
# long_answer_score
# short_answer start & end token
# short answer score
# yes_no_answer: None
def compute_predictions(args, id, candidates, token_map, logits, predictions):
    start_logits = predictions["start_predictions"][0]  # best_n_size
    start_positions = predictions["start_predictions"][1]  # best_n_size
    end_logits = predictions["end_predictions"][0]  # best_n_size
    end_positions = predictions["end_predictions"][1]  # best_n_size

    results = []

    for start_i, start_position in enumerate(start_positions):
        for end_i, end_position in enumerate(end_positions):
            if end_position < start_position:
                continue
            if end_position - start_position + 1 > args.max_answer_length:
                continue
            if token_map[int(start_position)] == -1:
                continue
            if token_map[int(end_position)] == -1:
                continue
            short_span_score = start_logits[start_i] + end_logits[end_i]
            cls_score = logits["start_logits"][0] + logits["end_logits"][0]
            score = short_span_score - cls_score

            start_span = token_map[int(start_position)]
            end_span = token_map[int(end_position)] + 1

            span = (start_span, end_span)

            answer_type = predictions["type_predictions"]

            results.append((score, span, short_span_score, cls_score, answer_type))

    # default empty prediction
    score = -10000
    short_span = (-1, -1)
    long_span = (-1, -1)

    # only want the highest ranked result
    # find corresponding long answer
    if results:
        short_prediction = sorted(results, key=lambda a: a[0], reverse=True)[0]
        score = short_prediction[0]
        short_span = short_prediction[1]
        for long_candidate in candidates:
            if (
                long_candidate["top_level"]
                and long_candidate["start_token"] <= short_span[0]
                and long_candidate["end_token"] >= short_span[1]
            ):
                long_span = (long_candidate["start_token"], long_candidate["end_token"])
                # only the first one is needed due to stride
                break

    summary = {
        "example_id": id,
        "long_answer_span": long_span,
        "long_score": score,
        "short_answer": short_span,
        "short_score": score,
        "yes_no_answer": "NONE",
    }

    return summary


# for each instance I have one prediction
# need to combine them so that each example_id has one prediction
def pick_best_prediction(summary_list):
    combined_predictions = {}
    best_predictions = {}

    for pred in summary_list:
        if pred["example_id"] not in combined_predictions.keys():
            combined_predictions[pred["example_id"]] = [pred]
        else:
            combined_predictions[pred["example_id"]].append(pred)

    for id, preds in combined_predictions.items():
        best_pred = sorted(preds, key=lambda a: a["short_score"], reverse=True)[0]
        best_predictions[id] = best_pred

    return best_predictions
