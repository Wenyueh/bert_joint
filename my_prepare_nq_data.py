import json
import gzip
import collections
import torch
import argparse

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
# add contexts to example


def get_answer(args):
    with gzip.GzipFile(args.train_data, "rb") as f:
        for line in f:
            data = json.loads(line)
            create_example(data)


def create_example(data):
    # still need to filter candidates by top_level, etc
    for candidate in data["long_answer_candidates"]:
        first_token_position = candidate["start_token"]
        first_token = data["document_tokens"][first_token_position]["token"]
        print(first_token)
        counts = collections.defaultdict(int)
        if first_token == "<Table>":
            counts["Table"] += 1
            candidate["type_and_position"] = ["Table", counts["Table"]]
        elif first_token == "<P>":
            counts["Paragraph"] += 1
            candidate["type_and_position"] = ["Paragraph", counts["Paragraph"]]
        elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
            counts["List"] += 1
            candidate["type_and_position"] = ["List", counts["List"]]
        elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
            counts["Other"] += 1
            candidate["type_and_position"] = ["Other", counts["Other"]]
        else:
            print("Unknown candidate type found: %s", first_token)

    annotation, candidate_idx, short_span_position = get_first_annotation(data)


def get_first_annotation(data):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data", type=str, default="data/dev/train/nq-train.jsonl.gz"
    )

    args = parser.parse_args()

    get_answer(args)
