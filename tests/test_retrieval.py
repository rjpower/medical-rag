import medrag.utils
import medrag.finetune
import pytest
import transformers


@pytest.fixture(scope="session", autouse=True)
def retriever():
    return medrag.utils.RetrievalSystem(
        retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./cache/test-db"
    )


@pytest.fixture(scope="session", autouse=True)
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct", padding_side="left"
    )


@pytest.fixture(scope="session", autouse=True)
def model():
    return transformers.AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16
    ).to("cuda")


benchmark_df = medrag.finetune.load_benchmark_dataset

def test_retrieval_lookup(retriever):
    results, scores = retriever.retrieve(
        "What is the most common cause of death in the United States?", k=10
    )

    assert len(results) == 10
    assert sorted(scores, reverse=True) == scores
    assert sorted(results[0].keys()) == ["content", "id", "title"]
    assert "heart" in results[0]["content"].lower(), results[0]["content"].lower()


def test_document_lookup(retriever):
    doc = retriever.get_document_by_id(
        "textbooks",
        "InternalMed_Harrison_644",
    )
    print(doc)
    assert "heart" in doc["content"].lower()


def test_prompt(tokenizer, model, benchmark_df):
    tokens = tokenizer(
        [medrag.finetune.build_prompt(benchmark_df.iloc[0])], return_tensors="pt"
    ).to("cuda")
    output = model.generate(
        **tokens,
        max_new_tokens=5,
        output_logits=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    output = tokenizer.decode(output["sequences"][0])
    print(output)
