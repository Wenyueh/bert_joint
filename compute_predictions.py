import gzip
import json
import os


# compute candidate_dict: id:candidates
# based on id, concatenate corresponding candidate list with predictions
# compute predictions with the corresponding id
# collect them and write them to json file
def compute_pred_dict(args, raw_results):
    candidate_dict = sorted(compute_candidate_dict(args).items())

    full_token_map_dict = compute_full_token_map_dict(args)

    # delete useless token_maps
    token_map_dict = {}
    for key in raw_results.keys():
        token_map_dict[key[0]] = full_token_map_dict[key]

    # leave only the first part of the key, the example_id used in original data
    new_raw_results = {}
    for k, v in raw_results.items():
        new_raw_results[k[0]] = v

    full_token_map_dict = sorted(token_map_dict.items())

    raw_results = sorted(raw_results.item())

    assert candidate_dict.keys() == token_map_dict.keys() == raw_results.keys()

    # based on raw results with ids and candidate dict
    candidate_prediction_results = {}
    for key in candidate_dict.keys():
        candidate_prediction_results[key] = {
            "candidates": candidate_dict[key],
            "token_map": token_map_dict[key],
            "raw_results": raw_results[key],
        }

    summary_list = []

    for id, candidate_prediction_result in candidate_prediction_results.items():
        candidates = candidate_prediction_result["candidates"]
        token_map = candidate_prediction_result["token_map"]
        logits = candidate_prediction_result["raw_results"]["logits"]
        predictions = candidate_prediction_result["raw_results"]

        summary = compute_predictions(
            args, id, candidates, token_map, logits, predictions
        )
        summary_list.append(summary)

    return summary_list


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
    for data in data_points:
        unique_index = tuple(data["unique_index"])  # list type is unhashable
        token_map = data["token_map"]
        token_map_dict[unique_index] = token_map
    return token_map_dict


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
