import numpy as np
import pandas as pd
import datasets

import torch
import transformers
import typing
import requests
import tqdm
import pathlib
import json

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import (
    evaluation,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
)

import medrag.utils

SYSTEM_PROMPT = (
    "You are a helpful medical expert, and your task is to answer a multi-choice medical question.\n"
    "Your responses will be used for research purposes only, so please have a definite answer.\n"
    "Answer using a only single letter, A, B, C, D, or E. Do not explain further.\n\n"
)


def build_prompt(record: typing.Dict, context: str = ""):
    question = record["question"]
    options = record["options"]
    options = "\n".join(f"{key}: {value}" for key, value in options.items())
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        + SYSTEM_PROMPT
        + "<|eot_id|><|start_header_id|>user<|end_header_id|>"
    )

    if context:
        prompt = prompt + "Here is some useful context for your question: " + context

    prompt = (
        prompt
        + f"Question: {question}\n\n"
        + options
        + "\n\nAnswer (A, B, C, D, or E): "
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )
    return prompt


def batch_predict(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    records: typing.List[typing.Dict],
    context: typing.List[str] = [""],
    batch_size=8,
    show_progress=False,
):
    # need to patch the LLama tokenizer to add a padding token
    tokenizer.pad_token = tokenizer.eos_token

    def _stop_on_eos(input_ids: torch.LongTensor, scores: torch.FloatTensor, **kw):
        return torch.all(input_ids == tokenizer.eos_token_id, dim=1)

    np.random.seed(42)
    torch.manual_seed(42)

    # split our input into batches and generate, then combine the results back together
    combined = {
        "text": [],
        "logits": [],
    }

    assert (
        len(records) == 1 or len(context) == 1
    ), "Only one of records or context should contain more than one element."

    count = max(len(records), len(context))
    batches = np.array_split(range(count), max(1, count // batch_size))
    if show_progress:
        batches = tqdm.tqdm(batches)

    for batch in batches:
        if len(records) == 1:
            prompts = [build_prompt(records[0], context[i]) for i in batch]
        else:
            prompts = [build_prompt(records[i], context[0]) for i in batch]
        tokens = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        output = model.generate(
            **tokens,
            max_new_tokens=2,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=[_stop_on_eos],
            output_logits=True,
            return_dict_in_generate=True,
        )
        logits = output["logits"][1].cpu()
        # strip input tokens from each output, because we're left padded, this is
        # the same for all sequences.
        sequences = output["sequences"].cpu()
        sequences = sequences[:, tokens["input_ids"].shape[1] :]
        text = tokenizer.batch_decode(sequences, clean_up_tokenization_spaces=True)
        for i in range(len(sequences)):
            combined["text"].append(text[i])
            combined["logits"].append(logits[i])

    results = []

    # get the answer token ids for computing logits/probits
    answer_tokens = tokenizer(["A", "B", "C", "D", "E"], return_tensors="pt")[
        "input_ids"
    ][:, 1]

    for i in range(count):
        logits = combined["logits"][i][answer_tokens]
        probits = torch.softmax(logits, dim=0)
        record = records[i] if len(records) > 1 else records[0]
        results.append(
            {
                "answer": combined["text"][i].strip(),
                "dataset": record["dataset"],
                "question": record["question"],
                "options": record["options"],
                "reference": record["answer"],
                "logits": logits.numpy(),
                "probits": probits.numpy(),
                "correct": record["answer"].strip() == combined["text"][i].strip(),
            }
        )
    return results


BENCHMARK_URL = (
    "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json"
)


def load_benchmark_dataset():
    pathlib.Path("./cache").mkdir(parents=True, exist_ok=True)

    if not pathlib.Path("./cache/benchmark.json").exists():
        r = requests.get(BENCHMARK_URL, stream=True)
        with open("./cache/benchmark.json", "wb") as f:
            pbar = tqdm.tqdm(unit="B", total=int(r.headers["Content-Length"]))
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    pbar.update(len(chunk))
                    f.write(chunk)

    with open("./cache/benchmark.json", "r") as f:
        benchmark = json.load(f)

    rows = []

    for dataset, data in benchmark.items():
        for k, v in data.items():
            question = v["question"]
            options = v["options"]
            answer = v["answer"]

            rows.append(
                {
                    "dataset": dataset,
                    "question": question,
                    "options": options,
                    "answer": answer,
                }
            )

    benchmark_df = pd.DataFrame(rows)
    return benchmark_df


def load_medqa():
    medqa = datasets.load_dataset("bigbio/med_qa")
    assert isinstance(medqa, datasets.DatasetDict)

    splits = []
    for split_name in medqa.keys():
        split = medqa[split_name]
        split = split.to_pandas()
        split = split.rename(columns={"answer": "answer_text"})
        split = split.rename(columns={"answer_idx": "answer"})

        # replace {'key': 'A', 'value': 'xyz' } in the options column with {'a': 'xyz', ...}
        def replace_keys_with_letters(options):
            return {opt["key"]: opt["value"] for opt in options}

        split["dataset"] = "medqa"
        split["options"] = split["options"].apply(replace_keys_with_letters)
        split["split"] = split_name
        splits.append(split)

    medqa = pd.concat(splits)
    return medqa


# Let's build a training dataset based on our document and a sample of our training data.


def build_finetune_dataset(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    retriever: medrag.utils.RetrievalSystem,
    training_set: pd.DataFrame,
    num_questions: int,
    docs_per_question: int,
):
    results = []
    sample_df = training_set.sample(num_questions, random_state=42)
    for _, record in tqdm.tqdm(sample_df.iterrows(), total=len(sample_df)):
        docs, scores = retriever.retrieve(record["question"], k=docs_per_question)
        naive_prediction = batch_predict(model, tokenizer, [record])[0]
        naive_probits = naive_prediction["probits"]
        predictions = batch_predict(
            model, tokenizer, [record], context=[doc["content"] for doc in docs]
        )

        for i, row in enumerate(predictions):
            doc_probits = row["probits"]
            diff_probits = doc_probits - naive_probits
            reference_idx = ord(row["reference"]) - ord("A")

            correct_diff = diff_probits[reference_idx]

            results.append(
                {
                    "question": row["question"],
                    "document": docs[i]["content"],
                    "reference": row["reference"],
                    "options": row["options"],
                    "diff": correct_diff,
                    "naive_answer": naive_prediction["answer"],
                    "naive_score": naive_probits[reference_idx],
                    "context_score": doc_probits[reference_idx],
                }
            )
    return pd.DataFrame(results)


def finetune_model(embedding_df):
    # Now let's fine tune our embedding model with our dataset using sentence transformers
    base_model_name = "ncbi/MedCPT-Query-Encoder"
    # define our model
    transformer_model = Transformer(base_model_name)
    pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), "cls")

    model = SentenceTransformer(
        modules=[transformer_model, pooling_model], device="cuda"
    )

    # make train and eval dataframes
    df = embedding_df
    df = df.rename(columns={"question": "sentence1", "document": "sentence2"})

    print(np.histogram(df["diff"]))

    # Adjust the score of the document based on whether it helped or hurt the model.
    def _remap_labels(row):
        if np.abs(row["diff"] > 0.5):
            return 1 if row["diff"] > 0 else -1

        if row["naive_score"] > 0.8:
            pass

        if np.abs(row["naive_score"]) < 0.3 and np.abs(row["diff"] < 0.3):
            return 0

        return 0

    df["label"] = df.apply(_remap_labels, axis="columns")
    df = df[df["label"] != 0]
    df = df[["sentence1", "sentence2", "label"]].reset_index(drop=True)

    train_df = df.iloc[: int(len(df) * 0.8)]
    eval_df = df.iloc[int(len(df) * 0.8) :]

    train_dataset = datasets.Dataset.from_pandas(train_df)
    print(train_dataset)

    train_loss = losses.CoSENTLoss(model=model)
    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        sentences1=eval_df["sentence1"].tolist(),
        sentences2=eval_df["sentence2"].tolist(),
        scores=eval_df["label"].tolist(),
    )

    training_args = SentenceTransformerTrainingArguments(
        output_dir="./cache/medcpt-embedding-model",
        eval_strategy="epoch",
        lr_scheduler_type="cosine",
        num_train_epochs=3,
        warmup_steps=10,
        weight_decay=1e-5,
        learning_rate=1e-3,
        per_device_train_batch_size=16,
        bf16=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset.take(128),
        loss=train_loss,
        evaluator=evaluator,
    )
    trainer.train()


def main():
    retriever = medrag.utils.RetrievalSystem(
        retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./cache/test-db"
    )

    medqa = load_medqa()

    if not pathlib.Path("./cache/embedding_ds.csv").exists():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", padding_side="left"
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16
        ).to("cuda")
        embedding_ds = build_finetune_dataset(
            tokenizer,
            model,
            retriever,
            medqa[medqa["split"] == "train"],
            num_questions=1000,
            docs_per_question=20,
        )
        embedding_ds.to_csv("./cache/embedding_ds.csv", index=False)
    else:
        embedding_ds = pd.read_csv("./cache/embedding_ds.csv")

    embedding_ds.sort_values(by="diff")
    print(embedding_ds)

    finetune_model(embedding_ds)


if __name__ == "__main__":
    main()
