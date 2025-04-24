import os

from BiEncoderWrapper import BiEncoderWrapper
from tasks import ChemBenchRetrieval, ChemRxivNC1
from mteb.overview import TASKS_REGISTRY
from mteb.model_meta import ModelMeta
from mteb.encoder_interface import PromptType
from functools import partial
import pandas as pd

import tqdm
import mteb
import argparse
from datetime import datetime
from statistics import mean
import time
import json
import re
import sys

TASKS_REGISTRY["ChemBenchRetrieval"] = ChemBenchRetrieval
TASKS_REGISTRY["ChemRxivNC1"] = ChemRxivNC1


def read_score(file):
    with open(file, "r") as f:
        data = json.load(f)
    scores = []
    for k in data["scores"].keys():
        scores.append(data["scores"][k][0]["main_score"])
    return mean(scores)


def extract_tag(path: str) -> str:
    match = re.search(r"(?:epoch|step)_\d+", path)
    return match.group() if match else None


def get_task_type(task_name):
    task = mteb.get_tasks(tasks=[task_name])[0]
    return task.metadata.type


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


def meta_builder(
    model_path: str,
    model_name: str,
    tokenizer_name: str,
    use_prefix: bool = True,
    seq_length: int = 512,
) -> ModelMeta:
    model_prompts = (
        {
            "Classification": "classification: ",
            "MultilabelClassification": "classification: ",
            "Clustering": "clustering: ",
            "PairClassification": "classification: ",
            "Reranking": "classification: ",
            "STS": "classification: ",
            "Summarization": "classification: ",
            PromptType.query.value: "search_query: ",
            PromptType.passage.value: "search_document: ",
        }
        if use_prefix
        else None
    )

    return ModelMeta(
        name=model_name,
        revision=None,
        release_date=None,
        languages=["eng-Latn"],
        license="cc-by-nc-4.0",
        framework=["Sentence Transformers", "PyTorch"],
        training_datasets=None,
        similarity_fn_name="cosine",
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=8192,
        embed_dim=768,
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        use_instructions=True,
        loader=partial(
            BiEncoderWrapper,
            model_name=model_path,
            tokenizer_name=tokenizer_name,
            seq_length=seq_length,
            model_prompts=model_prompts,
        ),
    )


def get_eval_paths(checkpoint_path: str) -> list[str]:
    folders = os.listdir(checkpoint_path)
    eval_paths = []
    for folder in folders:
        if "epoch" in folder and folder.endswith("model"):
            eval_paths.append(os.path.join(checkpoint_path, folder))
        if "step" in folder:
            eval_paths.append(os.path.join(checkpoint_path, folder, "model"))
    return eval_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cp", "--checkpoint_path", type=str, required=True)
    parser.add_argument("-op", "--output_path", type=str, default="benchmark")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_prefix", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["one-hop", "chemrxivp-ccby-1", "chemteb", "mteb"],
        default="chemteb",
    )

    args = parser.parse_args()

    if args.checkpoint_path in [
        "nomic-ai/nomic-embed-text-v1-unsupervised",
        "nomic-ai/nomic-embed-text-v1",
    ]:
        run_name = args.checkpoint_path
        eval_paths = [run_name]
        print(f"Running benchmark for {run_name}")
        run_name = args.checkpoint_path.replace("/", "__")
        results_path = os.path.join(args.output_path, run_name)
        print(f"Saving results to {results_path}")
        is_local = False
    else:
        run_name = args.checkpoint_path
        eval_paths = get_eval_paths(args.checkpoint_path)
        run_name = os.path.basename(os.path.normpath(args.checkpoint_path))
        print(f"Running benchmark for {run_name}")
        results_path = os.path.join(args.output_path, run_name)
        print(f"Saving results to {results_path}")
        is_local = True

    if (
        os.path.isdir(results_path)
        and os.listdir(results_path)
        and os.path.isfile(os.path.join(results_path, "raw_scores.csv"))
        and os.path.isfile(os.path.join(results_path, "mean_scores.csv"))
        and not args.overwrite
    ):
        print(
            f"Benchmark results already exist in {results_path}. Overwrite not set, exiting."
        )
        sys.exit(0)

    os.makedirs(results_path, exist_ok=True)

    if args.benchmark == "chemteb":
        bench = mteb.get_benchmark("ChemTEB")
    elif args.benchmark == "mteb":
        bench = mteb.get_benchmark("MTEB(eng, v1)")
    elif args.benchmark == "one-hop":
        bench = [ChemBenchRetrieval()]
    elif args.benchmark == "chemrxivp-ccby-1":
        bench = [ChemRxivNC1()]

    now = datetime.now()
    for eval_model in tqdm.tqdm(eval_paths):
        run_pos = extract_tag(eval_model) if is_local else run_name
        meta_builder_kwargs = {
            "model_path": eval_model,
            "model_name": run_pos,
            "tokenizer_name": args.tokenizer_name,
            "seq_length": args.seq_length,
        }
        if hasattr(args, "no_prefix") and args.no_prefix:
            meta_builder_kwargs["use_prefix"] = False
        model_meta = meta_builder(**meta_builder_kwargs)
        model = model_meta.loader()
        model.mteb_model_meta = model_meta

        evaluation = mteb.MTEB(tasks=bench)
        evaluation.run(
            model,
            encode_kwargs={"batch_size": args.batch_size},
            output_folder=results_path,
        )

    elapsed = datetime.now() - now
    elapsed_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed.total_seconds()))
    print(f"Completed benchmarking in {elapsed_formatted}")

    results_df = get_results(results_path)
    mean_df = results_df.mean()

    results_df.to_csv(os.path.join(results_path, "raw_scores.csv"))
    mean_df.to_csv(os.path.join(results_path, "mean_scores.csv"))
