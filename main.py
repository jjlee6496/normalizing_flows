import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from configs import RealNVPConfig, MAFConfig, GlowConfig
from models import RealNVP, MAF, Glow
from training.trainer import ImageFlowTrainer
from dataset import CelebADataset
from utils.setup import setup_experiment

from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                      choices=["realnvp", "maf", "glow"])
    parser.add_argument("--data_dir", type=str, default="data/celeba")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action='store_true')
    parser.add_argument("--no_tensorboard", action='store_true')
    parser.add_argument("--log_dir", type=str, default="logs")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load config
    if args.model == "realnvp":
        config = RealNVPConfig()
    elif args.model == "maf":
        config = MAFConfig()
    else:
        config = GlowConfig()
        
    # Update config with command line arguments
    config.data_dir = args.data_dir
    config.batch_size = args.batch_size
    config.n_epochs = args.n_epochs
    config.lr = args.lr
    config.seed = args.seed
    config.use_wandb = not args.no_wandb
    config.use_tenserboard = not args.no_tensorboard
    config.log_dir = args.log_dir
    
    # 2. Setup experiment (device, seed)
    device = setup_experiment(config)
    
    # 3. Create data module
    data_module = CelebADataset(
        data_dir=config.data_dir,
        train_batch_size=config.batch_size,
        val_batch_size=config.batch_size,
        patch_size=(config.image_size, config.image_size),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # 4. Create model based on config
    if config.model_type == "realnvp":
        model = RealNVP(
            input_dims=(config.in_channels, config.image_size, config.image_size),
            hidden_dims=config.hidden_dims,
            n_blocks=config.n_blocks,
            mask_type=config.mask_type,
            coupling_type=config.coupling_type,
        )
    elif config.model_type == "maf":
        model = MAF(
            input_size=config.in_channels * config.image_size * config.image_size,
            hidden_dims=config.hidden_dims,
            num_layers=config.num_layers,
            num_blocks=config.num_blocks
        )
    else:
        model = Glow(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            K=config.K,
            L=config.L
        )
    
    model = model.to(device)
    
    # 5. Create trainer
    trainer_module = ImageFlowTrainer(model, config)
    
    # 6. Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"{config.output_dir}/{config.model_type}/checkpoints",
            filename="{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            monitor="val/loss"
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]
    
    # 7. Setup logger
    loggers = []
    
    # Add WandB Logger if Enabled
    if config.use_wandb:
        wandb_logger = WandbLogger(
            project=config.project_name,
            name=f"{config.model_type}-run",
            config=vars(config),
            save_dir=config.output_dir
        )
        loggers.append(wandb_logger)
    
    # Add TensorBoard Logger if enabled
    if config.use_tensorboard:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_logger = TensorBoardLogger(
            save_dir=config.output_dir,
            name=config.model_type,
            version=f"run_{timestamp}",
            default_hp_metric=False
        )
        loggers.append(tensorboard_logger)
    
    # 8. Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator="auto",
        devices=1,
        logger=loggers if loggers else None,
        callbacks=callbacks,
        precision='16-mixed' if torch.cuda.is_available() else '32-true',  # Use mixed precision if available
        gradient_clip_val=config.gradient_clip_val,
        benchmark=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        profiler='simple',
        default_root_dir=config.log_dir
    )
    
    # 9. Train!
    trainer.fit(trainer_module, data_module)

if __name__ == "__main__":
    main()