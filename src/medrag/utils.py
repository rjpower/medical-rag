import collections
import typing
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

import os
import faiss
import json
import pathlib
import torch
import tqdm
import numpy as np

corpus_names = {
    "PubMed": ["pubmed"],
    "Textbooks": ["textbooks"],
    "StatPearls": ["statpearls"],
    "Wikipedia": ["wikipedia"],
    "MedCorp": ["pubmed", "textbooks", "statpearls", "wikipedia"],
}

retriever_names = {
    "BM25": ["bm25"],
    "Contriever": ["facebook/contriever"],
    "SPECTER": ["allenai/specter"],
    "MedCPT": ["ncbi/MedCPT-Query-Encoder"],
    "RRF-2": ["bm25", "ncbi/MedCPT-Query-Encoder"],
    "RRF-4": [
        "bm25",
        "facebook/contriever",
        "allenai/specter",
        "ncbi/MedCPT-Query-Encoder",
    ],
}


def load_model(model_name):
    if "contriever" in model_name:
        return SentenceTransformer(
            model_name, device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        transformer_model = Transformer(model_name)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), "cls")
        return SentenceTransformer(
            modules=[transformer_model, pooling_model],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )


def embed(dataset, corpus_name, index_dir, model_name, **kwarg):
    save_dir = pathlib.Path(index_dir) / "embedding"
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_name)
    model.eval()

    batch_size = 16384

    def _embed_batch(batch_id, texts):
        output_file = save_dir / f"{corpus_name}.{batch_id}.npy"
        if output_file.exists():
            return

        with torch.no_grad():
            embed_chunks = model.encode(texts, **kwarg)
            np.save(output_file, embed_chunks)

    with torch.no_grad():
        for batch_id in tqdm.tqdm(range(0, max(1, len(dataset) // batch_size))):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            texts = []
            title = dataset["title"][start:end]
            content = dataset["content"][start:end]
            id = dataset["id"][start:end]
            for i in range(0, len(title)):
                texts.append(
                    {
                        "title": title[i].as_py(),
                        "content": content[i].as_py(),
                        "id": id[i].as_py(),
                    }
                )

            if "specter" in model_name.lower():
                texts = [
                    model.tokenizer.sep_token.join([item["title"], item["content"]])
                    for item in texts
                ]
            elif "contriever" in model_name.lower():
                texts = [
                    ". ".join([item["title"], item["content"]])
                    .replace("..", ".")
                    .replace("?.", "?")
                    for item in texts
                ]
            elif "medcpt" in model_name.lower():
                texts = [[item["title"], item["content"]] for item in texts]

            _embed_batch(batch_id, texts)

    embed_chunks = model.encode([""], **kwarg)
    return embed_chunks.shape[-1]


def construct_index(
    corpus_name: str, index_dir: pathlib.Path, model_name, h_dim=768
) -> faiss.Index:

    with open(os.path.join(index_dir, "metadatas.jsonl"), "w") as f:
        f.write("")

    if "specter" in model_name.lower():
        index = faiss.IndexFlatL2(h_dim)
    else:
        index = faiss.IndexFlatIP(h_dim)

    offset = 0
    for fname in tqdm.tqdm(sorted((index_dir / "embedding").glob("*.npy"))):
        curr_embed = np.load(fname)
        index.add(curr_embed)
        with open(os.path.join(index_dir, "metadatas.jsonl"), "a+") as f:
            f.write(
                "\n".join(
                    [
                        json.dumps({"index": i, "source": corpus_name})
                        for i in range(offset, offset + len(curr_embed))
                    ]
                )
                + "\n"
            )
        offset += len(curr_embed)

    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    return index


class Retriever:

    def __init__(self, retriever_name, corpus_name, db_dir, **kwarg):
        self.name = retriever_name
        self.corpus_name = corpus_name

        self.db_dir = pathlib.Path(db_dir)
        if not self.db_dir.exists():
            self.db_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = load_dataset(f"MedRAG/{corpus_name}", num_proc=16)[
            "train"
        ].with_format("arrow")
        self.documents = {corpus_name: self.dataset}

        self.index_dir = (
            self.db_dir
            / "index"
            / self.corpus_name
            / self.name.replace("Query-Encoder", "Article-Encoder")
        )
        if os.path.exists(os.path.join(self.index_dir, "faiss.index")):
            self.index = faiss.read_index(os.path.join(self.index_dir, "faiss.index"))
            self.metadatas = [
                json.loads(line)
                for line in open(os.path.join(self.index_dir, "metadatas.jsonl"))
                .read()
                .strip()
                .split("\n")
            ]
        else:
            print(
                "[In progress] Embedding the {:s} corpus with the {:s} retriever...".format(
                    self.corpus_name,
                    self.name.replace("Query-Encoder", "Article-Encoder"),
                )
            )
            h_dim = embed(
                dataset=self.dataset,
                corpus_name=self.corpus_name,
                index_dir=self.index_dir,
                model_name=self.name.replace("Query-Encoder", "Article-Encoder"),
                **kwarg,
            )
            print(
                "[In progress] Embedding finished! The dimension of the embeddings is {:d}.".format(
                    h_dim
                )
            )
            self.index = construct_index(
                corpus_name=self.corpus_name,
                index_dir=self.index_dir,
                model_name=self.name.replace("Query-Encoder", "Article-Encoder"),
                h_dim=h_dim,
            )
            print("[Finished] Corpus indexing finished!")
            self.metadatas = [
                json.loads(line)
                for line in open(self.index_dir / "metadatas.jsonl")
                .read()
                .strip()
                .split("\n")
            ]
        self.embedding_function = load_model(self.name)
        self.embedding_function.eval()

    def retrieve(self, question, k=32, **kwarg):
        assert type(question) == str
        question = [question]

        if "bm25" in self.name.lower():
            res_ = [[]]
            hits = self.index.search(question[0], k=k)
            res_[0].append(np.array([h.score for h in hits]))
            indices = [
                {
                    "source": "_".join(h.docid.split("_")[:-1]),
                    "index": eval(h.docid.split("_")[-1]),
                }
                for h in hits
            ]
        else:
            with torch.no_grad():
                query_embed = self.embedding_function.encode(question, **kwarg)

            res_ = self.index.search(query_embed, k=k)
            indices = [self.metadatas[i] for i in res_[1][0]]

        docs = []
        for item in indices:
            source = item["source"]
            index = item["index"]
            docs.append(
                {
                    "source": source,
                    "index": index,
                }
            )

        scores = res_[0][0].tolist()
        return docs, scores


class RetrievalSystem:

    def __init__(
        self, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./cache"
    ):
        self.name = retriever_name
        self.corpus_name = corpus_name
        assert self.corpus_name in corpus_names
        if retriever_name in retriever_names:
            retrievers = retriever_names[retriever_name]
        else:
            retrievers = [retriever_name]

        self.retrievers = []
        for retriever in retrievers:
            self.retrievers.append(
                [
                    Retriever(retriever, corpus, db_dir)
                    for corpus in corpus_names[self.corpus_name]
                ]
            )

    def get_document_by_index(self, source, index):
        docs = self.retrievers[0][0].documents[source]
        document = docs[index]
        return {
            "id": document["id"][0].as_py(),
            "title": document["title"][0].as_py(),
            "content": document["content"][0].as_py(),
        }

    def get_document_by_id(self, source, id):
        docs = self.retrievers[0][0].documents[source]
        doc_ids = docs["id"]
        index = doc_ids.index(id).as_py()
        if index == -1:
            raise KeyError(
                "No document with id {:s} found in the corpus {:s}.".format(id, source)
            )
        return self.get_document_by_index(source, index)

    def retrieve(self, question, k=32, rrf_k=100):
        """
        Given `question`, return the top `k` relevant document ids from the corpus.
        """
        assert type(question) == str

        texts = []
        scores = []

        if "RRF" in self.name:
            k_ = max(k * 2, 100)
        else:
            k_ = k

        results = []
        for retriever_set in self.retrievers:
            for retriever in retriever_set:
                docs, scores = retriever.retrieve(question, k=k_)
                for doc, score in zip(docs, scores):
                    results.append((retriever.name, doc, score))

        return self.merge(results, k, rrf_k)

    def merge(
        self,
        results: typing.List[typing.Tuple[int, typing.Dict, float]],
        k=32,
        rrf_k=100,
    ):
        """
        Merge the texts and scores from different retrievers
        """
        merged = collections.defaultdict(float)
        for retriever, doc, score in results:
            merged[(doc["source"], doc["index"])] += score

        merged = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        # lookup doc content
        texts = []
        scores = []
        for (source, index), score in merged[:k]:
            texts.append(self.get_document_by_index(source, index))
            scores.append(score)
        return texts, scores
