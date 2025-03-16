import os

from BiEncoderWrapper import BiEncoderWrapper
from mteb.model_meta import ModelMeta
from mteb.encoder_interface import PromptType
from functools import partial
import pandas as pd

import tqdm
import mteb
import argparse
from datetime import datetime
import time
import json
import re


def read_score(file: str) -> float:
    with open(file, "r") as f:
        data = json.load(f)
    return data["scores"]["test"][0]["main_score"]


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
        rev = os.listdir(os.path.join(results_folder, model))[0]
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
    model_path: str, model_name: str, tokenizer_name: str, seq_length: int = 512
) -> ModelMeta:
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
            model_prompts={
                "Classification": "classification: ",
                "MultilabelClassification": "classification: ",
                "Clustering": "clustering: ",
                "PairClassification": "classification: ",
                "Reranking": "classification: ",
                "STS": "classification: ",
                "Summarization": "classification: ",
                PromptType.query.value: "search_query: ",
                PromptType.passage.value: "search_document: ",
            },
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
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    eval_paths = get_eval_paths(args.checkpoint_path)

    run_name = os.path.basename(os.path.normpath(args.checkpoint_path))
    print(f"Running benchmark for {run_name}")
    results_path = os.path.join("benchmark", run_name)
    print(f"Saving results to {results_path}")
    os.makedirs(results_path, exist_ok=True)

    bench = mteb.get_benchmark("ChemTEB")

    now = datetime.now()
    for eval_model in tqdm.tqdm(eval_paths):
        run_pos = extract_tag(eval_model)
        model_meta = meta_builder(
            eval_model, run_pos, args.tokenizer_name, args.seq_length
        )
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
