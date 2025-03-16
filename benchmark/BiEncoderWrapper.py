import os
import numpy as np
from typing import Any


import torch
import mteb


from transformers import AutoTokenizer
from mteb.models.wrapper import Wrapper
from mteb.encoder_interface import PromptType
from contrastors import BiEncoder, BiEncoderConfig


class BiEncoderWrapper(Wrapper):
    """
    A wrapper for a BiEncoder model that can be loaded from a local path.
    This class is compatible with the new version of MTEB and handles prompt prefixes.
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_name: str = "bert-base-uncased",
        seq_length: int = 512,
        rotary_scaling_factor: float | None = None,
        model_prompts: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        if os.path.exists(model_name):
            config = BiEncoderConfig.from_pretrained(model_name)
            if rotary_scaling_factor is not None:
                config.rotary_scaling_factor = rotary_scaling_factor
            self.model = BiEncoder.from_pretrained(model_name, config=config).to(
                torch.bfloat16
            )
        else:
            config = BiEncoderConfig(
                model_name=model_name, encoder=True, pooling="mean"
            )
            if rotary_scaling_factor is not None:
                config.rotary_scaling_factor = rotary_scaling_factor
            self.model = BiEncoder(config).to(torch.bfloat16)

        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.model_max_length = seq_length
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_prompts = self.validate_task_to_prompt_name(model_prompts)

    def to(self, device: torch.device) -> None:
        self.model.to(device)

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Encodes sentences by first adding a prefix (if applicable) based on the prompt type,
        then tokenizes and runs the model.
        """
        # Determine the prompt prefix based on the prompt type.
        # default to search_document if input_type and prompt_name are not provided
        prompt_name = (
            self.get_prompt_name(self.model_prompts, task_name, prompt_type)
            or PromptType.passage.value
        )

        prompt = self.model_prompts.get(prompt_name)
        task = mteb.get_task(task_name)

        # normalization not applied to classification
        # https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/eval/mteb_eval/eval_mteb.py#L172
        normalize = task.metadata.type not in (
            "Classification",
            "MultilabelClassification",
            "PairClassification",
            "Reranking",
            "STS",
            "Summarization",
        )

        if prompt:
            sentences = [prompt + sentence for sentence in sentences]

        batch_size = kwargs.get("batch_size", 32)
        embeddings = []
        self.model.to(self.device)

        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                encoded = self.tokenizer(
                    batch, padding=True, truncation=True, return_tensors="pt"
                )

                outputs = self.model(
                    **encoded.to(self.device),
                    normalize=normalize,
                )
                batch_embs = outputs["embedding"].cpu().float().numpy()
                embeddings.append(batch_embs)

        return np.concatenate(embeddings, axis=0)
