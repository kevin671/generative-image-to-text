import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
from llama.hf import LLaMATokenizer, LLaMAForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple


PREFIX_LENGTH = 5


class GenerativeImage2TextModel(nn.Module):
    def __init__(self, image_encoder, language_model, tokenizer, mapping_type="mlp"):
        super().__init__()
        self.image_encoder = image_encoder
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.visual_projection = create_projecton_layer(
            mapping_type, image_encoder.embed_dim, language_model.config.hidden_size
        )
        if hasattr(self.language_model, "model"):
            self.wte = self.language_model.model.get_input_embeddings()
        elif hasattr(self.language_model, "transformer"):
            self.wte = self.language_model.transformer.get_input_embeddings()
        else:
            raise NotImplementedError

    def image2token_embeddings(self, images):
        image_embeddings = self.image_encoder(
            images
        )  # RN50: torch.Size([50, 49, 2048]) for grid, torch.Size([50, 1024]) for non-grid
        if image_embeddings.shape[1] == 257:  # vit grid
            image_embeddings = image_embeddings[:, 1:, :]
        mapped_embeddings = self.visual_projection(image_embeddings)
        if mapped_embeddings.shape[-1] != self.language_model.config.hidden_size:
            # torch.Size([bs, 12800]) to torch.Size([bs, 10, 1280])
            mapped_embeddings = mapped_embeddings.view(-1, PREFIX_LENGTH, self.language_model.config.hidden_size)
        return mapped_embeddings

    def text2tokens(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", padding=True).input_ids.to(self.wte.weight.device).long()
        tokens_embeds = self.wte(tokens)
        return tokens, tokens_embeds

    def tokens2output(self, mapped_embeddings, tokens, tokens_embeds):
        inputs_embeds = torch.cat([mapped_embeddings, tokens_embeds], dim=1)
        labels = torch.cat(
            [torch.full((mapped_embeddings.shape[0], mapped_embeddings.shape[1]), -100, dtype=torch.long).to(tokens.device), tokens],
            dim=1,
        )
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
        )
        return outputs

    def forward(self, images, captions):
        mapped_embeddings = self.image2token_embeddings(images)
        tokens, tokens_embeds = self.text2tokens(captions)
        outputs = self.tokens2output(mapped_embeddings, tokens, tokens_embeds)
        return outputs

    def generate(self, image, prompt):
        mapped_embeddings = self.image2token_embeddings(image)
        _, tokens_embeds = self.text2tokens(prompt)
        outputs = self.language_model.generate(
            inputs_embeds=torch.cat([mapped_embeddings, tokens_embeds], dim=1),
            max_length=100,
            # do_sample=True,
            # top_p=0.9,
            # top_k=0,
            # temperature=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            early_stopping=True,
            num_beams=3,
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded


def get_git_model(args):
    image_encoder, _ = get_image_encoder(args)

    if "llama" in args.language_model:
        language_model = LLaMAForCausalLM.from_pretrained(args.language_model, device_map={'':torch.cuda.current_device()}, load_in_8bit=True)
        tokenizer = LLaMATokenizer.from_pretrained(args.language_model, add_eos_token=True)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        for param in language_model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)
        language_model.gradient_checkpointing_enable()  # reduce number of stored activations
        language_model.enable_input_require_grads()

        class CastOutputToFloat(nn.Sequential):
            def forward(self, x):
                return super().forward(x).to(torch.float32)

        language_model.lm_head = CastOutputToFloat(language_model.lm_head)

        """
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        language_model = get_peft_model(language_modell, config)
        print_trainable_parameters(language_model)
        """

    else:
        language_model = AutoModelForCausalLM.from_pretrained(args.language_model)
        tokenizer = AutoTokenizer.from_pretrained(args.language_model)
        for param in language_model.parameters():
            param.requires_grad = False
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    print("Language model memory footprint:", language_model.get_memory_footprint() / 1024**3, "GB")

    model = GenerativeImage2TextModel(
        image_encoder,
        language_model=language_model,
        tokenizer=tokenizer,
        mapping_type=args.mapping_type,
    )
    return model


def get_image_encoder(param):
    clip_model_type = param.image_encoder_type
    input_resolution = param.input_resolution
    model, processor = clip.load(clip_model_type, device="cpu", jit=False)
    model = model.train()
    ret = model.visual
    ret.to(torch.float32)
    ret.output_grid = param.output_grid
    ret.grid_after_ln = True
    ret.embed_dim = ret.embed_dim if ret.output_grid else ret.output_dim
    assert ret.input_resolution == input_resolution

    if isinstance(model.vision_layers, (tuple, list)) and ret.output_grid:
        del ret.attnpool

    return ret, processor


class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def create_projecton_layer(
    visual_projection_type,
    visual_feature_size,
    textual_feature_size,
):
    if visual_projection_type is None:
        return nn.Linear(visual_feature_size, textual_feature_size)
    elif visual_projection_type == "linearLn":
        return nn.Sequential(
            nn.Linear(visual_feature_size, textual_feature_size),
            nn.LayerNorm(textual_feature_size),
        )
    elif visual_projection_type == "mlp":
        return MLP(
            (visual_feature_size, (textual_feature_size * PREFIX_LENGTH) // 2, textual_feature_size * PREFIX_LENGTH)
        )
    elif visual_projection_type == "transformer":
        return nn.TransformerEncoderLayer(d_model=visual_feature_size, nhead=8, dim_feedforward=2048, dropout=0.1)
    else:
        raise NotImplementedError(visual_projection_type)
