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

from medrag.utils import RetrievalSystem


def evaluate(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset: datasets.Dataset,
    retriever: RetrievalSystem,
    batch_size: int = 8,
    min_retrieval_score: float = 0.0,
    device: str = "cuda:0",
):
    """Evaluate the model + retriever on the given dataset."""
    pass


def main():
    retriever = RetrievalSystem(
        retriever_name="./cache/medcpt-embedding-model",
        corpus_name="Textbooks",
        db_dir="./corpus",
    )
    results, scores = retriever.retrieve(
        "What is the most common cause of death in the United States?", k=10
    )

    results[1]
