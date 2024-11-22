from dataclasses import dataclass
from typing import List, Optional

@dataclass
class BaseConfig:
    # General
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "outputs"
    
    # Data
    data_dir: str = "data"
    image_size: int = 64
    in_channels: int = 3
    
    # Training
    batch_size: int = 32
    val_batch_size: int = 32
    n_epochs: int = 100
    lr: float = 1e-4
    lr_decay: float = 0.5
    lr_patience: int = 10
    weight_decay: float = 1e-5
    
    # Logging
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 1000
    log_dir: str = "logs"
    
    # Model general
    hidden_channels: int = 512
    hidden_dims: List[int] = (512, 512)
    dropout: float = 0.0
    use_batchnorm: bool = True
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "plateau"  # ['plateau', 'cosine', 'step']
    gradient_clip_val: float = 1.0
    
    # Validation
    n_samples_vis: int = 64
    n_interpolation_steps: int = 8
    
    # Metrics
    compute_fid: bool = True
    compute_inception: bool = True
    compute_ssim: bool = True
    
    # Loggers
    use_wandb: bool = True
    use_tensorboard: bool = True
    project_name: str = "normalizing-flows"
    entity: Optional[str] = None
    
    # Resources
    num_workers: int = 4
    pin_memory: bool = True