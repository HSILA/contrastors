import os
import time
import mteb
from tqdm import tqdm
from statistics import mean
import json
import pandas as pd
import torch
import argparse
from transformers import BitsAndBytesConfig
from mteb.encoder_interface import PromptType


from tasks import ChemRxivRetrieval
from sentence_transformers import SentenceTransformer
from mteb.overview import TASKS_REGISTRY
from dotenv import load_dotenv

load_dotenv()

# MODEL_PROMPTS = {
#     "Classification": "classification: ",
#     "MultilabelClassification": "classification: ",
#     "Clustering": "clustering: ",
#     "PairClassification": "classification: ",
#     "Reranking": "classification: ",
#     "STS": "classification: ",
#     "Summarization": "classification: ",
#     PromptType.query.value: "search_query: ",
#     PromptType.passage.value: "search_document: ",
# }


def read_score(file):
    with open(file, "r") as f:
        data = json.load(f)
    scores = []
    for k in data["scores"].keys():
        scores.append(data["scores"][k][0]["main_score"])
    return mean(scores)


def get_results(results_folder):
    models = os.listdir(results_folder)
    result = {}
    for model in models:
        model_path = os.path.join(results_folder, model)
        if not os.path.isdir(model_path):
            continue
        rev = os.listdir(model_path)[0]
        tasks = os.listdir(os.path.join(results_folder, model, rev))
        result[model] = {}
        for t in tasks:
            if t == "model_meta.json":
                continue
            score = read_score(os.path.join(results_folder, model, rev, t))
            task_name = os.path.splitext(t)[0]
            result[model][task_name] = score
    df = pd.DataFrame(result)
    return df


TASKS_REGISTRY["ChemRxivRetrieval"] = ChemRxivRetrieval


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")
    parser.add_argument("-o", "--output_path", help="Output path for results")
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument(
        "-b",
        "--bench",
        choices=["mteb-v2", "chemteb", "chemrxiv"],
        default="chemrxiv",
        help="Benchmark to run",
    )
    return parser.parse_args()


args = parse_args()
OUTPUT_PATH = args.output_path
BATCH_SIZE = args.batch_size

models = {
    "google-bert/bert-base-uncased": "86b5e0934494bd15c9632b12f734a8a67f723594",
    "allenai/scibert_scivocab_uncased": "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1",
    "intfloat/e5-small": "e272f3049e853b47cb5ca3952268c6662abda68f",
    "intfloat/e5-base": "b533fe4636f4a2507c08ddab40644d20b0006d6a",
    "intfloat/e5-large": "4dc6d853a804b9c8886ede6dda8a073b7dc08a81",
    "intfloat/e5-small-v2": "dca8b1a9dae0d4575df2bf423a5edb485a431236",
    "intfloat/e5-base-v2": "1c644c92ad3ba1efdad3f1451a637716616a20e8",
    "intfloat/e5-large-v2": "b322e09026e4ea05f42beadf4d661fb4e101d311",
    "intfloat/multilingual-e5-small": "fd1525a9fd15316a2d503bf26ab031a61d056e98",
    "intfloat/multilingual-e5-base": "d13f1b27baf31030b7fd040960d60d909913633f",
    "intfloat/multilingual-e5-large": "ab10c1a7f42e74530fe7ae5be82e6d4f11a719eb",
    "nomic-ai/nomic-bert-2048": "7710840340a098cfb869c4f65e87cf2b1b70caca",
    "nomic-ai/nomic-embed-text-v1.5": "b0753ae76394dd36bcfb912a46018088bca48be0",
    "nomic-ai/modernbert-embed-base": "5960f1566fb7cb1adf1eb6e816639cf4646d9b12",
    "nomic-ai/nomic-embed-text-v2-moe": "1066b6599d099fbb93dfcb64f9c37a7c9e503e85",
    "recobo/chemical-bert-uncased": "498698d28fcf7ce5954852a0444c864bdf232b64",
    "BAAI/bge-m3": "5617a9f61b028005a4858fdac845db406aefb181",
    "BAAI/bge-small-en": "2275a7bdee235e9b4f01fa73aa60d3311983cfea",
    "BAAI/bge-base-en": "b737bf5dcc6ee8bdc530531266b4804a5d77b5d8",
    "BAAI/bge-large-en": "abe7d9d814b775ca171121fb03f394dc42974275",
    "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "BAAI/bge-base-en-v1.5": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    "BAAI/bge-large-en-v1.5": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    "all-mpnet-base-v2": "84f2bcc00d77236f9e89c8a360a00fb1139bf47d",
    "multi-qa-mpnet-base-dot-v1": "3af7c6da5b3e1bea796ef6c97fe237538cbe6e7f",
    "all-MiniLM-L12-v2": "a05860a77cef7b37e0048a7864658139bc18a854",
    "all-MiniLM-L6-v2": "8b3219a92973c328a8e22fadcfa821b5dc75636a",
    "m3rg-iitd/matscibert": "ced9d8f5f208712c4a90f98a246fe32155b29995",
    "openai/text-embedding-ada-002": "2",
    "openai/text-embedding-3-small": "2",
    "openai/text-embedding-3-large": "2",
    "bedrock/cohere-embed-english-v3": "1",
    "bedrock/cohere-embed-multilingual-v3": "1",
    "bedrock/amazon-titan-embed-text-v2": "1",
    "bedrock/amazon-titan-embed-text-v1": "1",
    "answerdotai/ModernBERT-base": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
    "answerdotai/ModernBERT-large": "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
    "thenlper/gte-small": "17e1f347d17fe144873b1201da91788898c639cd",
    "thenlper/gte-base": "c078288308d8dee004ab72c6191778064285ec0c",
    "thenlper/gte-large": "4bef63f39fcc5e2d6b0aae83089f307af4970164",
    "NovaSearch/stella_en_1.5B_v5": "d03be74b361d4eb24f42a2fe5bd2e29917df4604",
    "Alibaba-NLP/gte-multilingual-base": "ca1791e0bcc104f6db161f27de1340241b13c5a4",
    "jinaai/jina-embeddings-v3": "215a6e121fa0183376388ac6b1ae230326bfeaed",
    "Qwen/Qwen3-Embedding-0.6B": "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418",
    "Qwen/Qwen3-Embedding-4B": "5cf2132abc99cad020ac570b19d031efec650f2b",
    "Qwen/Qwen3-Embedding-8B": "80946ea0efeac60523ec1a2cc5a65428a650007e",
}

REMOTE_CODE = [
    "jinaai/jina-embeddings-v3",
    "nomic-ai/nomic-embed-text-v1.5",
    "nomic-ai/nomic-embed-text-v2-moe",
    "nomic-ai/nomic-bert-2048",
    "nomic-ai/modernbert-embed-base",
    "Alibaba-NLP/gte-multilingual-base",
]


now = time.time()

for model_name, revision in tqdm(models.items()):
    print(f"Running benchmark for {model_name}")
    run_name = model_name.replace("/", "__")
    results_path = os.path.join(OUTPUT_PATH, run_name)

    current_batch_size = BATCH_SIZE
    if "Qwen" in model_name:
        current_batch_size = 4

    if args.bench == "mteb-v2":
        bench = mteb.get_benchmark("MTEB(eng, v2)")
    elif args.bench == "chemteb":
        bench = mteb.get_benchmark("ChemTEB")
    else:
        bench = [ChemRxivRetrieval()]

    evaluation = mteb.MTEB(tasks=bench)
    if "bge" in model_name and "en" in model_name:
        model = SentenceTransformer(model_name, revision=revision)
        model.gradient_checkpointing_enable()
    elif "Qwen3" in model_name:
        loader_kwargs = {
            "trust_remote_code": model_name in REMOTE_CODE,
            "model_kwargs": {
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                ),
            },
            "revision": revision,
        }
        model = mteb.get_model(model_name, **loader_kwargs)
    else:
        loader_kwargs = {
            "trust_remote_code": model_name in REMOTE_CODE,
            "revision": revision,
        }

        # Add model prompts for BASF-AI/Chembedding models
        # if model_name.startswith("BASF-AI/Chembedding"):
        #     loader_kwargs["model_prompts"] = MODEL_PROMPTS

        model = mteb.get_model(model_name, **loader_kwargs)

    evaluation.run(
        model,
        encode_kwargs={"batch_size": current_batch_size},
        output_folder=results_path,
    )

    results_df = get_results(results_path)
    mean_df = results_df.mean()

    results_df.to_csv(os.path.join(results_path, "raw_scores.csv"))
    mean_df.to_csv(os.path.join(results_path, "mean_scores.csv"))

print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - now))}")
