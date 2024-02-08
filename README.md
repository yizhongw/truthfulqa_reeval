# TruthfulQA Re-Evaluation

[TruthfulQA](https://github.com/sylinrl/TruthfulQA) is a widely-used benchmark for testing the truthfulness of language models. One metric that TruthfulQA uses is the [GPT Judge](https://github.com/sylinrl/TruthfulQA?tab=readme-ov-file#fine-tuning-gpt-3-for-evaluation), a GPT model finetuned for evaluating the truthfulness and informatiness of free-form generations. However, as OpenAI has deprecated the curie model, the finetuning procedure and the model itself are no longer available. This project aims to re-train the Judge model based on LLaMa and to compare the performance of the new model with the original Judge model.

## Data Setup

The original [TruthfulQA paper](https://arxiv.org/abs/2109.07958) analyzed how well the finetuned Judge model can generalize to the evaluation of a new model. They separate each model family, and use all other models for finetuning. They reported 88.4%-96.2% accuracy on the trufulness judge validation, and 88.9%-99.4% accuracy on the informativeness judge validation. See details in their appendix B.1.

However, their provided finetuning data ([truth](data/finetune_truth.jsonl) and [info](data/finetune_info.jsonl)) doesn't have the model information for each instance. So, we cannot reproduce this exact generalization experiment to specific model. We instead use a more relaxed setting: we group the examples by `question`, and then randomly select 10% of the outputs (which came from different models) as the dev set, and the rest outputs in the training set. We then finetune the Judge model on the training set, and evaluate the performance on the dev set. This is still a test on generalization to new models, but not specific to one model. You can reproduce this data split by running the `python src/split_data.py` script.

We will use this setup for do some analysis, and in the end we will use the entire provided dataset to finetune the final Judge model.

## Validation Details

To make the evaluation more reproducible, we decided to use open model (LLaMa2) instead of OpenAI's finetuning API. Following the original guidance on finetuning, we train the model for 5 epochs. We did a learning rate search, using both LLaMa2 7B and 13B. The training scripts can be found in the `scripts` folder.

## Validation Results

Here we present the validation results of the finetuned Judge model. The reported metrics here are accuracy (i.e., given a generation of a new model, how well the Judge model can classify it as truthful or informative, compared to the provided human labels). We also report the original Judge model's performance for comparison.

| Model | Truthful Accuracy | Informative Accuracy |
| --- | --- | --- |
| Original Judge | 0.884-0.962 | 0.889-0.994 |
| LLaMa2 7B Judge - LR 5e-6 | 0.945 | 0.957 |
| LLaMa2 7B Judge - LR 1e-5 | 0.944 | 0.954 |
| LLaMa2 7B Judge - LR 5e-5 | 0.943 | 0.952 |
| LLaMa2 13B Judge - LR 5e-6 | 0.948 |  -   |
| LLaMa2 13B Judge - LR 1e-5 | 0.950 |  -   |
| LLaMa2 13B Judge - LR 5e-5 | 0.950 |  -   |

You can find the logs on WanDB [here](https://wandb.ai/yizhongw/truthfulqa_reeval).

The validation here shows that we can achieve similar performance using LLaMa2 7B as the original Judge model. The difference of using LLaMa2 13B is not very significant, and the effect of leearning rate is almost negligible. We decide to use LLaMa2 7B for the final training, and use the learning rate 5e-6.

## Final Training and Model

Finally, we train two Judge models for truthfulness and informativenss classification on the entire provided data. We decided to use LLaMa 7B, as the performance is already good, and the original TruthfulQA paper also finetuned thier Judge model based on a 6.7B GPT. You can do this by running the following command:

```bash
./scripts/train_judge.sh
```

We have uploaded the final judge models to HuggingFace model hub. You can find the truthfulness model [here](https://huggingface.co/allenai/truthfulqa-truth-judge-llama2-7B) and the informativeness model [here](https://huggingface.co/allenai/truthfulqa-info-judge-llama2-7B).
