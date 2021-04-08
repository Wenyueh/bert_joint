import argparse
import json
import sys
import time
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, Dataset
# transformers version 3.0.2
from transformers import BertConfig, BertModel, get_linear_schedule_with_warmup

from model import Classification


class Logger:
    def __init__(self, path):
        self.log_path = path

    def log(self, string, newline=True):
        with open(self.log_path, "a") as f:
            f.write(string)
            if newline:
                f.write("\n")

        sys.stdout.write(string)
        if newline:
            sys.stdout.write("\n")
        sys.stdout.flush()


class FeatureData(Dataset):
    def __init__(self, args, data):
        self.data = data
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if args.train:
            return (
                torch.tensor(item["unique_index"]).long(),
                torch.tensor(item["input_ids"]).long(),
                torch.tensor(item["input_mask"]).long(),
                torch.tensor(item["segment_ids"]).long(),
                torch.tensor(item["start_position"]).long(),
                torch.tensor(item["end_position"]).long(),
                torch.tensor(item["answer_type"]).long(),
            )
        elif args.predict:
            return (
                torch.tensor(item["unique_index"]).long(),
                torch.tensor(item["input_ids"]).long(),
                torch.tensor(item["input_mask"]).long(),
                torch.tensor(item["segment_ids"]).long(),
                item["token_map"],
            )


def construct_optimizer(args, model, num_train_examples):
    no_weight_decay = ["LayerNorm.weight", "bias"]
    optimized_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(np in n for np in no_weight_decay)
            ],
            "weight_decay": 0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(np in n for np in no_weight_decay)
            ],
            "weight_decay": args.weight_decay,
        },
    ]
    # Implements linear decay of the learning rate.
    # default to be AdamW based on tensorflow, AdamWeightDecayOptimizer
    # parameters are using default
    optimizer = torch.optim.AdamW(optimized_parameters, lr=args.learning_rate)

    num_training_steps = int(
        args.epoch
        * num_train_examples
        / (args.batch_size * args.accumulate_gradient_steps)
    )
    num_warmup_steps = int(args.warm_up_proportion * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )

    return optimizer, scheduler


def initialize_by_squad_checkpoint(args):
    checkpoint = torch.load(args.squad_model)

    squad_pretrained_parameters = OrderedDict()

    """
    # initialize both the bert encoder and the qa_output layer
    for k, v in checkpoint.items():
        if "bert" in k:
            k = (
                "encoder." + k[5:]
            )  # change from bert.encoder.xxx to encoder.encoder.xxx
        if k != "qa_outputs.weight" and k != "qa_outputs.bias":
            squad_pretrained_parameters[k] = v
        elif k == "qa_outputs.weight":
            squad_pretrained_parameters["start_weights.weight"] = v[0].unsqueeze(0)
            squad_pretrained_parameters["end_weights.weight"] = v[1].unsqueeze(0)
        elif k == "qa_outputs.bias":
            squad_pretrained_parameters["start_weights.bias"] = v[0].unsqueeze(0)
            squad_pretrained_parameters["end_weights.bias"] = v[1].unsqueeze(0)

    config = BertConfig.from_json_file(args.bert_config)

    encoder = BertModel(config=config)

    model = Classification(args, encoder)
    model.load_state_dict(squad_pretrained_parameters, strict=False)
    """

    # initialize only the bert encoder
    # the qa_outputs weight and bias are not used based on the original code
    # since different names are used for the parameters: cls/nq/output_weights
    # also functions better if only use the encoder part
    for k, v in checkpoint.items():
        if "bert" in k:
            k = k[5:]  # change from bert.encoder.xxx to encoder.xxx
        if k != "qa_outputs.weight" and k != "qa_outputs.bias":
            squad_pretrained_parameters[k] = v

    config = BertConfig.from_json_file(args.bert_config)

    # initialize both the bert encoder and the qa_output layer
    # the qa_outputs weight and bias are not used based on the original code
    # since different names are used for the parameters: cls/nq/output_weights
    encoder = BertModel(config=config)
    encoder.load_state_dict(squad_pretrained_parameters)

    model = Classification(args, encoder)

    model.cuda()

    return model


# join examples with features and raw results:
# id, candidates both give, result and feature
# id, candidates, result (id, start_logits, end_logits, type_logits), features
# but I don't really think we need to concatenate with features here

# candidates_dict[example_id] = [long_answer_candidiates]
def candidate_dict(eval_data_dir):
    pass


# token_map_dict[example_id] = [token_map]
def token_map_dict(eval_feature_dir):
    pass


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
def compute_predictions(args, candidates, token_map, logits, predictions):
    max_answer_len = args.max_answer_len  # 30
    start_logits = predictions["start_positions"][0].tolist()  # batch * best_n_size
    start_positions = predictions["start_positions"][1].tolist()  # batch * best_n_size
    end_logits = predictions["end_positions"][0].tolist()  # batch * best_n_size
    end_positions = predictions["end_positions"][1].tolist()  # batch * best_n_size

    predictions = []

    for start_i, start_position in enumerate(start_positions):
        for end_i, end_position in enumerate(end_positions):
            if end_position < start_position:
                continue
            if end_position - start_position + 1 > max_answer_len:
                continue
            if token_map[start_position] == -1:
                continue
            if token_map[end_position] == -1:
                continue
            short_span_score = start_logits[start_i] + end_logits[end_i]
            cls_score = logits["start_logits"][0] + logits["end_logits"][1]
            score = short_span_score - cls_score
            span = (token_map[start_position], token_map[end_position] + 1)

            answer_type = predictions["type_predictions"]

            predictions.append((score, span, answer_type))


# compute candidate_dict: id:candidates
# based on id, concatenate corresponding candidate list with predictions
# compute predictions with the corresponding id
# collect them and write them to json file
def compute_pred_dict(args, raw_results):
    candidate_dict = candidate_dict(args.eval_dir)

    # load features from json file
    token_map_dict = token_map_dict(args.eval_feature_dir)

    # based on raw results with ids and candidate dict
    candidate_prediction_pairs = None

    summary_list = []

    for candidate_prediction_pair in candidate_prediction_pairs:
        id = None
        candidates = candidate_dict[id]
        token_map = token_map_dict[id]
        logits = raw_results[id]["logits"]
        predictions = raw_results[id]["predictions"]

        summary = compute_predictions(args, candidates, token_map, logits, predictions)
        summary_list.append(summary)

    with open(args.eval_result_dir, "w") as f:
        json.load(f, summary_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="preprocessed_data_dev_train.json"
    )
    parser.add_argument(
        "--squad_model",
        type=str,
        default="squad_model/pytorch_model.bin",
        help="The bert model is pretrained on SQuAD 2.0. Initialize the Bert Model by squad_pretrained_model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best_model.pt",
        help="location where the trained model will be saved",
    )
    parser.add_argument("--bert_config", type=str, default="bert_config.json")
    parser.add_argument("--train", type=bool, default=True)
    # training related
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--warm_up_proportion", type=float, default=0.1)
    parser.add_argument("--accumulate_gradient_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    # logging
    parser.add_argument("--log_path", type=str, default="dev.log")
    parser.add_argument("--logging_step", type=int, default=10)
    # eval
    parser.add_argument("--eval_mode", type=bool, default=False)
    parser.add_argument("--best_n_size", type=int, default=20)
    parser.add_argument("--max_answer_length", type=int, default=30)
    parser.add_argument("--eval_dir", type=str, default="data/dev/dev")

    args = parser.parse_args()

    logger = Logger(args.log_path)
    logger.log(str(args))

    with open(args.data_dir, "r") as f:
        data = json.load(f)

    # initialize the model by squad
    model = initialize_by_squad_checkpoint(args)

    # load data
    FeatureData = FeatureData(args, data)
    train_loader = DataLoader(FeatureData, batch_size=args.batch_size, shuffle=True)

    # optimizer & scheduler
    num_train_examples = len(FeatureData)
    optimizer, scheduler = construct_optimizer(args, model, num_train_examples)

    num_steps = 0
    used_time = 0
    for epoch in range(args.epoch):
        one_epoch_start_time = time.time()
        for i, data in enumerate(train_loader):
            start_time = time.time()

            loss_value = 0

            # input to the model
            input_ids = data[1].cuda()
            input_mask = data[2].cuda()
            segment_ids = data[3].cuda()
            start_positions = data[4].cuda()
            end_positions = data[5].cuda()
            types = data[6].cuda()

            model.train()

            loss, logits, predictions = model(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_positions=start_positions,
                end_positions=end_positions,
                types=types,
            )

            loss.backward()

            loss_value += loss.item()

            # update the model
            if (i + 1) % args.accumulate_gradient_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                num_steps += 1

                torch.cuda.synchronize()
                end_time = time.time()
                used_time += end_time - start_time

                if num_steps % args.logging_step == 0:
                    logger.log(
                        "From step {} to step {}, which is epoch {} batch {}, the total time used is {}, the averaged loss value is {}".format(
                            num_steps - args.logging_step,
                            num_steps,
                            epoch,
                            i,
                            used_time,
                            loss_value / args.logging_step,
                        )
                    )
                    used_time = 0
                    loss_value = 0
