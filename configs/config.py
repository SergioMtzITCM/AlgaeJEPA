from dataclasses import dataclass
from typing import Optional

@dataclass
class MicroViTConfig:
    model: str = "S1"
    image_size: int = 224
    num_channels: int = 3
    num_classes: int = None

    # Stability
    layerscale_value: float = 1e-5

@dataclass
class MobileNetConfig:
    model: str = "V2"
    image_size: int = 224
    num_channels: int = 3
    num_classes: int = None

@dataclass
class ResNetConfig:
    image_size: int = 224
    num_channels: int = 3
    num_classes: int = None

# --- Configuration ---
@dataclass
class BaseConfig:
    hidden_size: int = 256
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    intermediate_size: int = 512
    hidden_act: str = "silu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    rope_theta: float = 100.0
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    qkv_bias: bool = True
    num_classes: int = None

    # Stability
    qk_norm: bool = True
    qk_norm_bias: bool = False
    qk_norm_affine: bool = False # If True, learns gamma/beta in QKNorm
    layerscale_value: float = 1e-5
    drop_path_prob: float = 0.0 # Stochastic depth
    use_swiglu: bool = True

    # Augmented Coordinates (RoPE 2D)
    pos_embed_shift: Optional[float] = None
    pos_embed_jitter: Optional[float] = None
    pos_embed_rescale: float = 2.0

    # Masking
    prediction_ratio: float = 0.6

    # Predictor
    num_predictor_layers: int = 4
    use_abs_pos: bool = False

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
            )