import argparse
import gzip
import json
import os
import sys
import time
from collections import OrderedDict
from os.path import isdir

import torch
import torch.nn as nn
from numpy.lib.function_base import _parse_gufunc_signature
from torch.utils.data import DataLoader, Dataset
# transformers version 3.0.2
from transformers import BertConfig, BertModel, get_linear_schedule_with_warmup

from compute_predictions import (compute_candidate_dict,
                                 compute_full_token_map_dict,
                                 compute_predictions)
from model import Classification
from my_prepare_nq_data import FeatureData
from nq_eval import (compute_best_f1, compute_plain_f1, load_gold_labels,
                     load_prediction_labels, score_predictions)


def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


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
    optimizer = torch.optim.Adam(optimized_parameters, lr=args.learning_rate)

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

    if args.dp:
        model = nn.DataParallel(model)

    model.cuda()

    return model


def load_model(args, predict):
    config = BertConfig.from_json_file(args.bert_config)

    encoder = BertModel(config=config)

    model = Classification(args, encoder)

    if args.dp:
        model = nn.DataParallel(model)

    trained_model = torch.load(args.model)

    modified_trained_model = OrderedDict()
    for k, v in trained_model.items():
        if k[:4] == "bert":  # for bert_base
            newk = k.replace("bert", "encoder")
            modified_trained_model[newk] = v
        elif k[:7] == "module.":  # for bert_large
            newk = k.replace("module.", "")
            modified_trained_model[newk] = v
        else:
            modified_trained_model[k] = v

    model.load_state_dict(modified_trained_model, strict=False)

    model.cuda()

    if predict:
        model.eval()
    else:
        model.train()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="fix random seed")
    parser.add_argument("--bert_type", type=str, default="bert_large_uncased")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="preprocessed_data_full_train.json",
        help="preprocessed training data directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large_best_model.pt",
        help="location where the trained model will be saved or loaded",
    )
    parser.add_argument("--continue_training", type=bool, default=False)
    parser.add_argument("--train", type=bool, default=False)
    # training related
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--warm_up_proportion", type=float, default=0.1)
    # actual batch_size 32
    parser.add_argument("--accumulate_gradient_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--save_checkpoint_steps", type=int, default=1000)
    # eval
    parser.add_argument("--evaluation", type=bool, default=True)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--best_n_size", type=int, default=10)
    parser.add_argument("--max_answer_length", type=int, default=30)
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default="data/full/dev",
        help="the original data for evaluation, non-preprocessed",
    )
    parser.add_argument(
        "--eval_feature_dir",
        type=str,
        default="preprocessed_data_full_dev.json",
        help="the preprocessed evaluation data directory",
    )
    parser.add_argument(
        "--eval_result_dir",
        type=str,
        default="large_prediction_result.json",
        help="the directory to save predictions of the model on eval dataset",
    )
    parser.add_argument("--long_non_null_answer_threshold", type=int, default=2)
    parser.add_argument("--short_non_null_answer_threshold", type=int, default=2)
    # logging
    parser.add_argument("--log_path", type=str, default="dev.log")
    parser.add_argument("--logging_step", type=int, default=100)
    parser.add_argument("--eval_logging_steps", type=int, default=1000)

    parser.add_argument("--gpu", type=str, default="0,1")

    args = parser.parse_args()

    if args.bert_type == "bert_base_uncased":
        args.bert_config = "bert_base_config.json"
        args.squad_model = "squad_bert_base"
    if args.bert_type == "bert_large_uncased":
        args.bert_config = "bert_large_config.json"
        args.squad_model = "squad_bert_large/pytorch_model.bin"

    set_seed(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logger = Logger(args.log_path)
    logger.log(str(args))

    # data parallel
    args.dp = torch.cuda.device_count() > 1

    if args.train:

        with open(args.data_dir, "r") as f:
            data = json.load(f)
            # just in case there's only one data point and not as a one-element list
            if isinstance(data, dict):
                data = [data]

        # load data
        TrainFeatureData = FeatureData("train", data)
        train_loader = DataLoader(
            TrainFeatureData, batch_size=args.batch_size, shuffle=True
        )

        if args.continue_training:
            # loading trained model, keep training
            model = load_model(args, False)
        else:
            # initialize the model by squad
            logger.log("Loading SQuAD pretrained model...")
            model = initialize_by_squad_checkpoint(args)
            logger.log("Finished loading SQuAD pretrained model.")

        # optimizer & scheduler
        num_train_examples = len(TrainFeatureData)
        optimizer, scheduler = construct_optimizer(args, model, num_train_examples)

        # train on training dataset
        num_steps = 0
        used_time = 0
        model.zero_grad()
        for epoch in range(args.epoch):
            loss_value = 0
            one_epoch_start_time = time.time()
            for i, data in enumerate(train_loader):
                start_time = time.time()

                # input to the model
                unique_index = data[0].cuda()
                input_ids = data[1].cuda()
                input_mask = data[2].cuda()
                segment_ids = data[3].cuda()
                start_positions = data[4].cuda()
                end_positions = data[5].cuda()
                types = data[6].cuda()

                loss, logits, predictions, _ = model(
                    unique_index=unique_index,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_positions=start_positions,
                    end_positions=end_positions,
                    types=types,
                )

                if args.dp:
                    loss = loss.mean()

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

                    # the model does not evaluate the training dataset
                    # save checkpoint every xxx steps
                    # since the model does not evaluate on training,
                    # it not necessarily save the best model
                    if num_steps != 0 and num_steps % args.save_checkpoint_steps == 0:
                        torch.save(model.state_dict(), args.model)
                        logger.log("Save model at step {}".format(num_steps))

        torch.save(model.state_dict(), args.model)
        logger.log("Save model at step {}".format(num_steps))

    # do prediction on dev dataset and evaluation here
    if args.evaluation:

        logger.log("Loading preprocessed development data set...")

        with open(args.eval_feature_dir, "r") as f:
            data = json.load(f)
            # if data is only one datapoint not in the form of a list of len 1
            if isinstance(data, dict):
                data = [data]

        # load evaluation data
        EvalFeatureData = FeatureData("evaluation", data)
        eval_loader = DataLoader(
            EvalFeatureData, batch_size=args.eval_batch_size, shuffle=False
        )

        # loading them in order to compute prediction spans
        logger.log(
            "Loading unique_index: candidate_list from {}...".format(args.eval_data_dir)
        )
        candidate_dict = compute_candidate_dict(args.eval_data_dir)

        logger.log(
            "Loading unique_index: token_map from {}...".format(args.eval_feature_dir)
        )
        token_map_dict = compute_full_token_map_dict(args.eval_feature_dir)

        # load trained model
        logger.log("Loading trained model for evaluation...")
        model = load_model(args, args.evaluation)
        logger.log("Finished loading trained model for evaluation.")

        summary_list = []
        for batch_num, batch in enumerate(eval_loader):
            # input to the model
            unique_index = batch[0].cuda()
            input_ids = batch[1].cuda()
            input_mask = batch[2].cuda()
            segment_ids = batch[3].cuda()
            start_positions = None
            end_positions = None
            types = None

            _, logits, predictions, unique_index = model(
                unique_index=unique_index,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_positions=start_positions,
                end_positions=end_positions,
                types=types,
            )

            for i in range(unique_index.size(0)):
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

                # compile the prediction result
                # find corresponding long_answer_candidates
                orig_id = unique_index[i].tolist()[0]
                for k, one_candidates in candidate_dict.items():
                    if k == orig_id:
                        candidates = one_candidates

                # find corresponding token_map
                full_id = unique_index[i].tolist()
                for k, one_token_map in token_map_dict.items():
                    if k == tuple(full_id):
                        token_map = one_token_map

                summary = compute_predictions(
                    args, orig_id, candidates, token_map, one_logit, one_prediction
                )

                # at most one answer for each 512 input_ids
                # could be multiple answers for each example_id
                summary_list.append(summary)

            if batch_num != 0 and batch_num % args.eval_logging_steps == 0:
                logger.log(
                    "Running model on development data set for {} batches.".format(
                        batch_num
                    )
                )

        with open(args.eval_result_dir, "w") as f:
            json.dump(summary_list, f)

        # compute evaluation on long answers and short answers
        gold_label = load_gold_labels(args)
        # load prediction data and transform to nq_lable structure
        # id: nq_label
        pred_label = load_prediction_labels(args)

        long_answer_stats, short_answer_stats = score_predictions(
            args, gold_label, pred_label
        )

        (
            long_precision,
            long_recall,
            long_f1,
            short_precision,
            short_recall,
            short_f1,
        ) = compute_plain_f1(long_answer_stats, short_answer_stats)

        logger.log("*" * 20)
        logger.log("Print plain score result")
        logger.log("Long answer precision is: {}".format(long_precision))
        logger.log("Long answer recall is: {}".format(long_recall))
        logger.log("Long answer f1 is: {}".format(long_f1))
        logger.log("Short answer precision is: {}".format(short_precision))
        logger.log("Short answer recall is: {}".format(short_recall))
        logger.log("Short answer f1 is: {}".format(short_f1))

        (
            (best_long_f1, best_long_precision, best_long_recall, best_long_threshold,),
            _,
        ) = compute_best_f1(long_answer_stats, targets=[0.5, 0.75, 0.9])

        (
            (
                best_short_f1,
                best_short_precision,
                best_short_recall,
                best_short_threshold,
            ),
            _,
        ) = compute_best_f1(short_answer_stats, targets=[0.5, 0.75, 0.9])

        logger.log("*" * 20)
        logger.log("Print optimized score result")
        logger.log("Long answer score threshold: {}".format(best_long_threshold))
        logger.log("Long answer precision is: {}".format(best_long_precision))
        logger.log("Long answer recall is: {}".format(best_long_recall))
        logger.log("Long answer f1 is: {}".format(best_long_f1))
        logger.log("Short answer score threshold: {}".format(best_short_threshold))
        logger.log("Short answer precision is: {}".format(best_short_precision))
        logger.log("Short answer recall is: {}".format(best_short_recall))
        logger.log("Short answer f1 is: {}".format(best_short_f1))
