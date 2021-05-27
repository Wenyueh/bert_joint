import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from utils import truncated_normal_

# initialize bert model (input_ids, input_mask, segment_ids)
# add two dense layers + biases, since we have start/end to predict
# output logits
# add answer_type weight and bias
# compute answer_type_logits
# return (start logits, end_logits, type_logits)
#
# train:
# if has checkpoint: load checkpoint
# loss function for start/end position: log_softmax
# loss function for type: log_softmax
# total loss is the average of the sum of losses
# optimizer: linear_warm_up


class Classification(nn.Module):
    def __init__(self, args, encoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.hidden_size = self.encoder.config.hidden_size
        self.start_weights = nn.Linear(self.hidden_size, 1, bias=True)
        self.end_weights = nn.Linear(self.hidden_size, 1, bias=True)
        self.answer_type_number = 5
        self.type_weights = nn.Linear(
            self.hidden_size, self.answer_type_number, bias=True
        )

        self.loss_fct = nn.CrossEntropyLoss()

        self.start_weights.weight.data = truncated_normal_(self.start_weights.weight)
        self.end_weights.weight.data = truncated_normal_(self.end_weights.weight)
        self.type_weights.weight.data = truncated_normal_(self.type_weights.weight)

        self.start_weights.bias.data.zero_()
        self.end_weights.bias.data.zero_()
        self.type_weights.bias.data.zero_()

    def forward(
        self,
        unique_index,
        input_ids,
        input_mask,
        segment_ids,
        start_positions,
        end_positions,
        types,
    ):

        sequence_hidden = self.encoder(input_ids, input_mask, segment_ids)[
            0
        ]  # B, seq_len, hidden_size
        cls_hidden = self.encoder(input_ids, input_mask, segment_ids)[
            1
        ]  # B, hidden_size

        start_logits = self.start_weights(sequence_hidden).squeeze()  # B, seq_len
        end_logits = self.end_weights(sequence_hidden).squeeze()  # B, seq_len
        type_logits = self.type_weights(cls_hidden).squeeze()  # B, 5

        if len(start_logits.size()) == 1:
            start_logits = start_logits.unsqueeze(0)
            end_logits = end_logits.unsqueeze(0)
            type_logits = type_logits.unsqueeze(0)
        logits = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "type_logits": type_logits,
        }

        # if train, compute loss
        loss = None
        if (
            torch.is_tensor(start_positions)
            and torch.is_tensor(end_positions)
            and torch.is_tensor(types)
        ):
            start_loss = self.loss_fct(start_logits, start_positions)
            end_loss = self.loss_fct(end_logits, end_positions)
            type_loss = self.loss_fct(type_logits, types)
            loss = (start_loss + end_loss + type_loss) / 3.0

        # compute prediction
        # return best 20 positions ignoring CLS, sorted from high to low based on logits
        # tested right
        start_predictions = torch.topk(
            start_logits[:, 1:], self.args.best_n_size, dim=1
        )
        end_predictions = torch.topk(end_logits[:, 1:], self.args.best_n_size, dim=1)
        type_predictions = torch.argmax(type_logits, dim=1)

        predictions = {
            "start_predictions": start_predictions,
            "end_predictions": end_predictions,
            "type_predictions": type_predictions,
        }

        return (loss, logits, predictions, unique_index)
