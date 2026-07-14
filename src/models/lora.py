import torch
import torch.nn as nn
import torch.nn.init as init


class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=1.0):
        super().__init__()
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        has_bias = original_linear.bias is not None

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.original = original_linear
        for param in self.original.parameters():
            param.requires_grad = False

        device = original_linear.weight.device
        self.A = nn.Parameter(torch.empty(in_features, rank, device=device))
        self.B = nn.Parameter(torch.empty(rank, out_features, device=device))
        if has_bias:
            self.lora_bias = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter("lora_bias", None)

        init.kaiming_uniform_(self.A, a=5 ** 0.5)
        init.zeros_(self.B)

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.A) @ self.B * self.scale
        if self.lora_bias is not None:
            lora_out = lora_out + self.lora_bias
        return original_out + lora_out


def inject_lora_to_emotion_head(model, rank=4, alpha=1.0):
    original = model.emotion_head
    model.emotion_head = LoRALinear(original, rank=rank, alpha=alpha)
    return model


def remove_lora_from_emotion_head(model):
    if isinstance(model.emotion_head, LoRALinear):
        model.emotion_head = model.emotion_head.original
    return model


def lora_state_dict(model):
    if not isinstance(model.emotion_head, LoRALinear):
        return {}
    lora = model.emotion_head
    return {
        "emotion_head.A": lora.A.data.clone(),
        "emotion_head.B": lora.B.data.clone(),
        "emotion_head.lora_bias": lora.lora_bias.data.clone() if lora.lora_bias is not None else None,
    }


def load_lora_state_dict(model, state_dict):
    if not isinstance(model.emotion_head, LoRALinear):
        model = inject_lora_to_emotion_head(model)
    lora = model.emotion_head
    lora.A.data.copy_(state_dict["emotion_head.A"])
    lora.B.data.copy_(state_dict["emotion_head.B"])
    if lora.lora_bias is not None and state_dict.get("emotion_head.lora_bias") is not None:
        lora.lora_bias.data.copy_(state_dict["emotion_head.lora_bias"])
    return model
