import random
import json
import re

random.seed(42)

for data_file in ["data/finetune_info.jsonl", "data/finetune_truth.jsonl"]:
    examples_grouped_by_question = {}
    with open(data_file, "r") as f:
        for line in f:
            example = json.loads(line)
            # extract the part between Q: and A:
            question = re.search(r'Q: (.*)\nA:', example["prompt"]).group(1).strip()
            assert question
            if question not in examples_grouped_by_question:
                examples_grouped_by_question[question] = []
            examples_grouped_by_question[question].append(example)

    base_name = data_file.split("/")[-1].split(".")[0]
    with open(f"data/{base_name}_train.jsonl", "w") as f_train, open(f"data/{base_name}_dev.jsonl", "w") as f_dev:
        for question, examples in examples_grouped_by_question.items():
            random.shuffle(examples)
            n = len(examples)
            n_train = int(0.9 * n)
            for example in examples[:n_train]:
                f_train.write(json.dumps(example) + "\n")
            for example in examples[n_train:]:
                f_dev.write(json.dumps(example) + "\n")