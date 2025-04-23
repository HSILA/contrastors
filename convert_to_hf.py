import os
from argparse import ArgumentParser

from contrastors.models.biencoder import BiEncoder, BiEncoderConfig
from contrastors.models.dual_encoder import DualEncoder, DualEncoderConfig
from contrastors.models.huggingface import (
    NomicBertConfig,
    NomicBertForPreTraining,
    NomicVisionModel,
)
from huggingface_hub import HfApi, HfFolder, upload_file, hf_hub_download


CUSTOM_CODES = [
    "./src/contrastors/models/huggingface/configuration_hf_nomic_bert.py",
    "./src/contrastors/models/huggingface/modeling_hf_nomic_bert.py",
]

VOCAB_FILES = [
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "1_Pooling/config.json",
]


def push_modeling_files(repo_id):
    token = HfFolder.get_token()
    api = HfApi()
    existing_files = api.list_repo_files(
        repo_id=repo_id, repo_type="model", token=token
    )

    for file_path in CUSTOM_CODES:
        file_name = os.path.basename(file_path)
        if file_name not in existing_files:
            print(f"Uploading {file_name} to repository...")
            upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="model",
                token=token,
                commit_message=f"Add {file_name}",
            )


def push_tokenizer_files(target_repo_id, files_list):
    """
    For each file in extra_files, download it from source_repo_id
    and push it to target_repo_id if it doesn't already exist.
    """
    token = HfFolder.get_token()
    api = HfApi()
    existing_files = api.list_repo_files(
        repo_id=target_repo_id, repo_type="model", token=token
    )

    for file_name in files_list:
        if file_name not in existing_files:
            print(
                f"Uploading tokenizer file {file_name} from 'nomic-ai/nomic-embed-text-v1' to repository..."
            )
            # Download file from the source repository
            local_path = hf_hub_download(
                repo_id="nomic-ai/nomic-embed-text-v1",
                filename=file_name,
                repo_type="model",
                token=token,
            )
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=file_name,
                repo_id=target_repo_id,
                repo_type="model",
                token=token,
                commit_message=f"Add extra file {file_name}",
            )
            os.remove(local_path)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--biencoder", action="store_true")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--use_temp_dir", action="store_true")
    parser.add_argument("--push_tokenizer", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.biencoder:
        config = BiEncoderConfig.from_pretrained(args.ckpt_path)
        model = BiEncoder.from_pretrained(args.ckpt_path, config=config)
        model = model.trunk
        model.config.__delattr__("_name_or_path")
        model.config.auto_map = {
            "AutoConfig": "configuration_hf_nomic_bert.NomicBertConfig",
            "AutoModel": "modeling_hf_nomic_bert.NomicBertModel",
            "AutoModelForMaskedLM": "modeling_hf_nomic_bert.NomicBertForPreTraining",
        }
    elif args.vision:
        NomicBertConfig.register_for_auto_class()
        NomicVisionModel.register_for_auto_class("AutoModel")
        config = DualEncoderConfig.from_pretrained(args.ckpt_path)
        model = DualEncoder.from_pretrained(args.ckpt_path, config=config)
        vision = model.vision
        hf_config = NomicBertConfig(**model.vision.trunk.config.to_dict())
        model = NomicVisionModel(hf_config)

        state_dict = vision.state_dict()
        state_dict = {k.replace("trunk.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        config = NomicBertConfig.from_pretrained(args.ckpt_path)
        if not hasattr(config, "auto_map") or config.auto_map is None:
            config.auto_map = {
                "AutoConfig": "configuration_hf_nomic_bert.NomicBertConfig",
                "AutoModel": "modeling_hf_nomic_bert.NomicBertModel",
                "AutoModelForMaskedLM": "modeling_hf_nomic_bert.NomicBertForPreTraining",
                "AutoModelForSequenceClassification": "modeling_hf_nomic_bert.NomicBertForSequenceClassification",
                "AutoModelForMultipleChoice": "modeling_hf_nomic_bert.NomicBertForMultipleChoice",
                "AutoModelForQuestionAnswering": "modeling_hf_nomic_bert.NomicBertForQuestionAnswering",
                "AutoModelForTokenClassification": "modeling_hf_nomic_bert.NomicBertForTokenClassification",
            }
        model = NomicBertForPreTraining.from_pretrained(args.ckpt_path, config=config)

    model.push_to_hub(args.model_name, private=args.private)
    push_modeling_files(args.model_name)

    if args.push_tokenizer and args.biencoder:
        push_tokenizer_files(args.model_name, VOCAB_FILES)
