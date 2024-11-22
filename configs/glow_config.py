from .base_config import BaseConfig
from dataclasses import dataclass

@dataclass
class GlowConfig(BaseConfig):
    model_type: str = 'glow'
    hidden_channels: int = 512
    K: int = 32 # number of flow steps
    L: int = 3 # number of scales
    coupling_type: str = 'affine'