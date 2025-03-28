from transformers.configuration_utils import PretrainedConfig


class BiEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        model_name="EleutherAI/pythia-1b",
        projection_dim=None,
        logit_scale=1 / 0.07,
        use_fused_kernels=True,
        pooling="last",
        nomic_encoder=False,
        freeze=False,
        trainable_logit_scale=False,
        hamming=False,
        pretrained=False,
        gradient_checkpointing=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=["Wqkv", "out_proj", "fc11", "fc12"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.logit_scale = logit_scale
        self.trainable_logit_scale = trainable_logit_scale
        self.use_fused_kernels = use_fused_kernels
        self.pooling = pooling
        self.nomic_encoder = nomic_encoder
        self.freeze = freeze
        self.hamming = hamming
        self.pretrained = pretrained
        self.gradient_checkpointing = gradient_checkpointing
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
