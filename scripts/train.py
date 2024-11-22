# scripts/train.py
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from configs import RealNVPConfig, MAFConfig, GlowConfig
from models import RealNVP, MAF, Glow
from training.trainer import ImageFlowTrainer
from dataset import MyCelebA

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["realnvp", "maf", "glow"])
    parser.add_argument("--data_path", type=str, default="data/celeba")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Config 설정
    if args.model == "realnvp":
        config = RealNVPConfig()
    elif args.model == "maf":
        config = MAFConfig()
    else:
        config = GlowConfig()
        
    config.data_dir = args.data_path
    config.batch_size = args.batch_size
    config.n_epochs = args.n_epochs
    config.lr = args.lr
    
    # 2. 모델 생성
    if args.model == "realnvp":
        model = RealNVP(
            input_dims=(3, config.image_size, config.image_size),
            hidden_dims=config.hidden_dims,
            n_blocks=config.n_blocks,
            mask_type=config.mask_type,
            coupling_type=config.coupling_type
        )
    elif args.model == "maf":
        model = MAF(
            input_size=3 * config.image_size * config.image_size,
            hidden_dims=config.hidden_dims,
            num_layers=config.num_layers,
            num_blocks=config.num_blocks
        )
    else:
        model = Glow(
            in_channels=3,
            hidden_channels=config.hidden_channels,
            K=config.K,
            L=config.L
        )
    
    # 3. Data Module 생성
    data_module = MyCelebA(
        data_path=config.data_dir,
        train_batch_size=config.batch_size,
        val_batch_size=config.batch_size,
        patch_size=(config.image_size, config.image_size),
        num_workers=4,
        pin_memory=True
    )
    
    # 4. Flow Trainer 생성
    flow_trainer = ImageFlowTrainer(model, config)
    
    # 5. PyTorch Lightning Trainer 설정
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=WandbLogger(project="normalizing-flows"),
        callbacks=[
            ModelCheckpoint(
                dirpath=f"{config.output_dir}/checkpoints",
                filename="{epoch:02d}-{val_loss:.2f}",
                save_top_k=3,
                monitor="val/loss"
            ),
            LearningRateMonitor()
        ]
    )
    
    # 6. 학습 실행
    trainer.fit(flow_trainer, data_module)

if __name__ == "__main__":
    main()