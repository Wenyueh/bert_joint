import json
import gzip
import collections
import torch
import argparse
import enum
from transformers import BertTokenizer

# get_example
# for each dictionary extracted:
# (1) add type and position for each dict and their long_answer_candidates
# (2) find the first annotation, the corresponding cand_idx, short answer span positions
# (3) obtain question
# (4) obtain answer containing
# (a) anno_idx
# (b) span_text with html
# (c) span start within the long answer string
# (d) span end within the long answer string
# (e) type
# (5) candidate_list: (a) empty_candidate (b) cand_itr: id, type, (c) text_map (d) text
# assemble example: document_title, example_id, question, answer, has_correct_answer
# add candidates to example


def get_examples(args):
    with gzip.GzipFile(args.train_data, "rb") as f:
        for line in f:
            data = json.loads(line)
            example = create_example(args, data)
            yield example


# tested long, short, yes, unknown
def create_example(args, data):
    # first, filter the candidates
    filtered_candidates_list = filtered_candidates(args, data)
    # add type for each candidate
    # tested correct
    counts = collections.defaultdict(int)
    for candidate in filtered_candidates_list:
        # add type_and_position to each candidate
        first_token_position = candidate["start_token"]
        first_token = data["document_tokens"][first_token_position]["token"]
        if first_token == "<Table>":
            counts["Table"] += 1
            candidate["type_and_position"] = "[Table=%s]" % counts["Table"]
        elif first_token == "<P>":
            counts["Paragraph"] += 1
            candidate["type_and_position"] = "[Paragraph=%s]" % counts["Paragraph"]
        elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
            counts["List"] += 1
            candidate["type_and_position"] = "[List=%s]" % counts["List"]
        elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
            counts["Other"] += 1
            candidate["type_and_position"] = "[Other=%s]" % counts["Other"]
        else:
            print("Unknown candidate type found: %s", first_token)

    # tested correct
    annotation, candidate_idx, short_span_positions = get_first_annotation(data)

    # (4) obtain answer containing
    # (a) anno_idx
    # (b) span_text with html
    # (c) span start within the long answer string
    # (d) span end within the long answer string
    # (e) type
    answer = {
        # index in the original long_answer_candidates instead of the filtered version
        "annotation_index": candidate_idx,
        "span_text": "",
        "span_start": -1,
        "span_end": -1,
        "type": "",
    }

    # define answer type
    if annotation:
        # if short
        # tested correct
        if not short_span_positions == (-1, -1):
            start_position = short_span_positions[0]
            end_position = short_span_positions[1]
            answer["type"] = "short"
            candidate_text = get_candidate_text(
                data, data["long_answer_candidates"], candidate_idx
            ).text
            answer["span_text"] = candidate_text[start_position:end_position]
            answer["span_start"] = start_position
            answer["span_end"] = end_position

        elif annotation["long_answer"]["candidate_index"] > -1:
            answer["span_text"] = get_candidate_text(
                data, data["long_answer_candidates"], candidate_idx
            ).text
            answer["span_start"] = 0
            answer["span_end"] = len(answer["span_text"])

            if annotation["yes_no_answer"] == "NONE":
                answer["type"] = "long"

            elif annotation["yes_no_answer"] == "YES":
                answer["type"] = "yes"

            elif annotation["yes_no_answer"] == "NO":
                answer["type"] = "no"

        # find new candidate idx in filtered candidate list
        gold_start_token = annotation["long_answer"]["start_token"]
        for idx, cand in enumerate(filtered_candidates_list):
            if cand["start_token"] == gold_start_token:
                candidate_idx = idx
    else:
        answer["type"] = "unknown"
        answer["span_start"] = 29
        answer["span_end"] = 29  # length of [ContextID=-1] [NoLongAnswer]

        # aggregate candidate list
        # add an empty candidate at the beginning
        # tested correct
    offset = 0
    candidates_text = ["[ContextId=%d] %s" % (-1, "[NoLongAnswer]")]
    offset += len(candidates_text[-1]) + 1
    candidates_map = [-1, -1]
    candidate_indices = [-1]
    for idx, cand in enumerate(filtered_candidates_list):
        if idx < args.max_candidates - 1:
            candidate_indices.append(idx)
            # insert the number of candidates
            # the type of candidates
            # and the number this type occurs so far
            candidates_text.append(
                "[ContextId=%d] %s"
                % (
                    data["long_answer_candidates"].index(cand),
                    cand["type_and_position"],
                )
            )
            # +1 because of ' '
            offset += len(candidates_text[-1]) + 1
            if idx == candidate_idx:
                answer["span_start"] += offset
                answer["span_end"] += offset

            candidates_text.append(
                get_candidate_text(data, filtered_candidates_list, idx).text
            )
            offset += len(candidates_text[-1]) + 1
            # use [-1, -1] to correspond to [contextid=] and [type_and_position]
            candidates_map.extend([-1, -1])
            candidates_map.extend(
                get_candidate_text(data, filtered_candidates_list, idx).text_positions
            )
        else:
            break

    candidates_text = " ".join(candidates_text)

    example = {
        "name": data["document_title"],
        "id": str(data["example_id"]),
        "question": data["question_text"],
        "answer": answer,
        "has_correct_candidate": candidate_idx in candidate_indices,
        "candidates": candidates_text,
        "candidates_map": candidates_map,
    }

    return example


# tested correct
def get_candidate_text(data, candidates_list, candidate_idx):
    TextSpan = collections.namedtuple("TextSpan", "text_positions text")

    if candidate_idx < 0 or candidate_idx >= len(candidates_list):
        return TextSpan([], "")

    start_position = candidates_list[candidate_idx]["start_token"]
    end_position = candidates_list[candidate_idx]["end_token"]
    token_positions = []
    span = []
    for i in range(start_position, end_position):
        token = data["document_tokens"][i]
        if token["html_token"] == False:
            word = token["token"].replace(" ", "")
            token_positions.append(i)
            span.append(word)

    span = " ".join(span)

    return TextSpan(token_positions, span)


# tested correct
def get_first_annotation(data):

    sorted_annotations = sorted(
        [x for x in data["annotations"] if has_long_answer(x)],
        key=lambda a: a["long_answer"]["candidate_index"],
    )

    if sorted_annotations:
        annotation = sorted_annotations[0]

        cand_index = annotation["long_answer"]["candidate_index"]

        if annotation["short_answers"]:
            short_start_token = annotation["short_answers"][0]["start_token"]
            short_end_token = annotation["short_answers"][-1]["end_token"]
            long_start_token = annotation["long_answer"]["start_token"]
            start_offset = compute_offset(data, long_start_token, short_start_token)
            end_offset = compute_offset(data, long_start_token, short_end_token)

            return annotation, cand_index, (start_offset, end_offset)

        else:
            return annotation, cand_index, (-1, -1)
    else:
        return None, -1, (-1, -1)


# tested correct
def has_long_answer(annotation):
    if (
        not annotation["long_answer"]["start_token"] == -1
        and not annotation["long_answer"]["end_token"] == -1
    ):
        return True
    else:
        return False


# tested to be correct
def compute_offset(data, token_idx_1, token_idx_2):
    tokens = data["document_tokens"]
    char_offset = 0
    for index in range(token_idx_1, token_idx_2):
        token = tokens[index]
        if not token["html_token"]:
            token_length = len(tokens[index]["token"].replace(" ", ""))
            char_offset += token_length + 1

    return char_offset


# tested correct
def filtered_candidates(args, data):
    filtered_candidates = []
    for idx, cand in enumerate(data["long_answer_candidates"]):
        if get_candidate_text(
            data, data["long_answer_candidates"], idx
        ).text.strip() == "" or (
            args.skip_non_top_level_candidates and not cand["top_level"]
        ):
            continue
        else:
            filtered_candidates.append(cand)

    return filtered_candidates


# split candidates, for each char, identify which token it belongs to
# collect information about an example:
# (a) example_id, which is int
# (b) question id
# (c) question text
# (d) doc_tokens
# (e) doc_tokens_map
# (f) answer: (extractive) answer_type (AnswerType class), text, start
# to check whether is is extractive:
# (g) start and end position for token
# convert the example to inputfeatures:
# tokenize evey thing: doc_tokens, [Q], query tokens
# collect into inputfeatures:
# id, example_idx, doc_span_idx, token_to_orig_map, input_ids, input_mask,
# segment_ids, start_position, end_position, answer_text, type


class AnswerType(enum.IntEnum):
    UNKNOWN = 0
    YES = 1
    NO = 2
    SHORT = 3
    LONG = 4


class NQExample:
    def __init__(
        self,
        example_id,
        q_id,
        question,
        doc_tokens,
        doc_tokens_map,
        answer=None,
        start_position=None,
        end_position=None,
    ):
        self.example_id = example_id
        self.q_id = q_id
        self.question = question
        self.doc_tokens = doc_tokens
        self.doc_tokens_map = doc_tokens_map
        self.answer = None
        self.start_position = None
        self.end_position = None


class process_example:
    def __init__(self, args, tokenizer):
        self.train = args.train
        self.tokenizer = tokenizer

    def process(self, example):
        nq_example = return_nq_example(self.train, example)
        input_feature = example2feature(nq_example)


def return_nq_example(train, example):
    doc_tokens = example["candidates"].split()
    char_to_token_offset = []
    for idx, token in enumerate(doc_tokens):
        for w in token:
            char_to_token_offset.append(idx)
        if not idx == len(doc_tokens) - 1:
            char_to_token_offset.append(idx)
        else:
            continue
    assert len(char_to_token_offset) == len(example["candidates"])

    question_id = "{}".format(example["id"])
    question_text = example["question"]

    answer = None
    start_token_position = None
    end_token_position = None

    if train:

        Answer = collections.namedtuple("Answer", "type text offset")

        answer = example["answer"]

        def obtain_answer_type(answer):
            if answer["type"] == "unknown":
                answer_type = AnswerType.UNKNOWN
            elif answer["type"] == "yes":
                answer_type = AnswerType.YES
            elif answer["type"] == "no":
                answer_type = AnswerType.NO
            elif answer["type"] == "short":
                answer_type = AnswerType.SHORT
            elif answer["type"] == "long":
                answer_type = AnswerType.LONG

            return answer_type

        answer_type = obtain_answer_type(answer)

        candidates_text = example["candidates"]

        def obtain_answer_text_and_start(candidates_text, answer):
            start_position = answer["span_start"]
            end_position = answer["span_end"]
            if answer["type"] == "unknown":
                start_position = 0
                end_position = 1

            return candidates_text[start_position:end_position], start_position

        answer_text, start_position = obtain_answer_text_and_start(
            candidates_text, answer
        )

        answer = Answer(answer_type, answer_text, start_position)

        # token start position
        start_token_position = char_to_token_offset[start_position]
        end_token_position = char_to_token_offset[start_position + len(answer_text)]

        # check whether the provided answer text is in the actual doc_tokens
        actual_text = " ".join(doc_tokens[start_token_position:end_token_position])

        cleaned_answer_text = " ".join(answer_text.split())
        if actual_text.find(cleaned_answer_text) == -1:
            print(
                "Could not find answer '%s' in actual text '%s'"
                % (cleaned_answer_text, actual_text)
            )
            return []
        else:
            print(
                "Found answer '%s' in actual text '%s'"
                % (cleaned_answer_text, actual_text)
            )

        returned_example = NQExample(
            example["id"],
            question_id,
            question_text,
            doc_tokens,
            example["candidates_map"],
            answer,
            start_token_position,
            end_token_position,
        )

    return returned_example


def example2feature(nq_example):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data", type=str, default="data/dev/train/nq-train.jsonl.gz"
    )
    parser.add_argument("--max_candidates", type=int, default=50)
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--skip_non_top_level_candidates", type=bool, default=True)

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    processor = process_example(args, tokenizer)
    instances = []
    for example in get_examples(args):
        e = return_nq_example(example)
        input_feature = example2feature(e)
        instances.append(input_feature)
