import pytorch_lightning as pl
import torch
import wandb
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from utils.metrics import (
    InceptionStatistics, 
    calculate_ssim, 
    calculate_fid, 
    calculate_inception_score
)
from visualization.plotting import ImageFlowVisualizer

class ImageFlowTrainer(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        
        # 출력 디렉토리 설정 및 생성
        self.output_dir = Path(config.output_dir)
        self.exp_dir = self.output_dir / f"{config.model_type}_experiment"
        self.image_dir = self.exp_dir / "images"
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.log_dir = self.exp_dir / "logs"
        
        # 디렉토리들 생성
        for dir_path in [self.image_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup visualizer and metrics
        self.visualizer = ImageFlowVisualizer(model, self.device)
        if config.compute_inception:
            self.inception = InceptionStatistics(self.device)
            # inception 모델을 정식 서브모듈로 등록
            self.register_module('inception_model', self.inception.model)
        else:
            self.inception = None
        self.validation_step_outputs = []
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
    def _log_images(self, tag, images, global_step=None):
        """통합된 이미지 로깅 함수"""
        if global_step is None:
            global_step = self.global_step
            
        # 이미지를 파일로 저장
        save_path = self.image_dir / f"{tag}_{global_step}.png"
        save_image(images, save_path)
        
        # WandB 로깅
        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log({
                f"{tag}": wandb.Image(str(save_path)),
                "global_step": global_step
            })
        
        # TensorBoard 로깅
        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
            if len(images.shape) == 4:  # 배치 이미지인 경우
                self.logger.experiment.add_images(
                    f"{tag}",
                    images,
                    global_step
                )
            else:  # 단일 이미지 그리드인 경우
                self.logger.experiment.add_image(
                    f"{tag}",
                    images,
                    global_step
                )
    
    def _log_figure(self, tag, figure, global_step=None):
        """통합된 figure 로깅 함수"""
        if global_step is None:
            global_step = self.global_step
            
        # Figure를 파일로 저장
        save_path = self.image_dir / f"{tag}_{global_step}.png"
        figure.savefig(save_path, bbox_inches='tight', dpi=300)
        
        # WandB 로깅
        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log({
                f"{tag}": wandb.Image(str(save_path)),
                "global_step": global_step
            })
        
        # TensorBoard 로깅
        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
            self.logger.experiment.add_figure(
                f"{tag}",
                figure,
                global_step
            )
        
        plt.close(figure)
    
    def _log_metrics(self, metrics_dict, step=None):
        """통합된 메트릭 로깅 함수"""
        # 모든 메트릭이 tensor인 경우 detach하고 float로 변환
        processed_metrics = {}
        for k, v in metrics_dict.items():
            if isinstance(v, torch.Tensor):
                processed_metrics[k] = v.detach().float().cpu().item()
            else:
                processed_metrics[k] = v
        
        # 로거별 로깅
        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log({
                **processed_metrics,
                "global_step": step if step is not None else self.global_step
            })
        
        # Lightning의 내장 로깅 사용 (TensorBoard 포함)
        self.log_dict(processed_metrics, on_step=step is not None, on_epoch=step is None)
    
    def training_step(self, batch):
        images = batch[0].to(self.device)
        
        # Forward pass
        z_list, log_det = self.model(images)
        log_likelihood = sum(self.model.base_dist(z).log_prob(z).sum([1, 2, 3]) for z in z_list)
        loss = -(log_likelihood + log_det).mean()
        
        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Periodic visualization - wandb가 있을 때만
        if self.global_step % self.config.log_interval == 0:
            if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
                with torch.no_grad():
                    # Get reconstructions
                    recon_grid = self.visualizer.visualize_reconstructions(images)
                    self._log_images("train/reconstructions", recon_grid)
        
        return loss

    def validation_step(self, batch):
        images = batch[0].to(self.device)
        
        # Forward pass
        z_list, log_det = self.model(images)
        log_likelihood = sum(self.model.base_dist(z).log_prob(z).sum([1, 2, 3]) for z in z_list)
        loss = -(log_likelihood + log_det).mean()
        
        with torch.no_grad():
            # Get reconstructions and samples
            recon_images, _ = self.model(z_list, reverse=True)
            samples, _ = self.model.sample(len(images))
            
             # Calculate metrics
            metrics = {
                'val_loss': loss,
                'val_ssim': calculate_ssim(images, recon_images) if self.config.compute_ssim else None,
            }
            
            # Store inception activations if enabled
            if self.config.compute_inception and self.inception is not None:
                with torch.amp.autocast('cuda', enabled=False):
                    metrics.update({
                        'real_acts': self.inception.get_activations(images),
                        'fake_acts': self.inception.get_activations(samples)
                    })
            
            metrics['image'] = images
            self.validation_step_outputs.append(metrics)
        
        return metrics

    def on_validation_epoch_end(self):
        # 기본 메트릭 계산
        val_metrics = {}
        
        # Loss와 SSIM 계산
        val_metrics['val/loss'] = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        if self.config.compute_ssim:
            val_metrics['val/ssim'] = torch.stack([x['val_ssim'] for x in self.validation_step_outputs]).mean()
        
        # Inception 메트릭 계산
        if self.config.compute_inception and self.inception is not None:
            real_acts = torch.cat([x['real_acts'] for x in self.validation_step_outputs])
            fake_acts = torch.cat([x['fake_acts'] for x in self.validation_step_outputs])
            
            fid = calculate_fid(real_acts, fake_acts)
            is_mean, is_std = calculate_inception_score(fake_acts)
            
            val_metrics.update({
                'val/fid': fid,
                'val/inception_score': is_mean,
                'val/inception_score_std': is_std
            })
        
        # 메트릭 로깅
        self._log_metrics(val_metrics)
        
        # 시각화
        if len(self.validation_step_outputs) > 0:
            images = self.validation_step_outputs[0]['image']
            
            # 이미지 저장 경로 설정 및 디렉토리 생성
            val_img_dir = self.image_dir / "val"
            val_img_dir.mkdir(parents=True, exist_ok=True)
            base_path = val_img_dir / f"flow_visualization_{self.current_epoch}"
            
            # Flow 시각화 - 각 이미지 저장 및 개별 로깅
            self.visualizer.plot_flow(images, str(base_path))
            
            # 각 이미지에 대해 개별적으로 로깅
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log({
                    "val/reconstruction": wandb.Image(str(base_path) + "_reconstruction.png"),
                    "val/linear_interpolation": wandb.Image(str(base_path) + "_linear.png"),
                    "val/spherical_interpolation": wandb.Image(str(base_path) + "_spherical.png"),
                    "epoch": self.current_epoch
                })
            
            # TensorBoard 로깅
            if isinstance(self.logger, pl.loggers.TensorBoardLogger):
                for img_type in ['reconstruction', 'linear', 'spherical']:
                    img_path = f"{base_path}_{img_type}.png"
                    if Path(img_path).exists():
                        img = plt.imread(img_path)
                        self.logger.experiment.add_image(
                            f"val/{img_type}",
                            img.transpose(2, 0, 1),  # HWC -> CHW
                            self.current_epoch
                        )
        
        # Clear stored outputs
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # Optimizer 설정
        if self.config.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config.lr,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        # Scheduler 설정
        if self.config.scheduler.lower() == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.config.lr_decay,
                patience=self.config.lr_patience,
                verbose=True
            )
            scheduler_config = {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "frequency": 1
            }
        elif self.config.scheduler.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.n_epochs,
                eta_min=self.config.lr * 0.01
            )
            scheduler_config = {
                "scheduler": scheduler,
                "frequency": 1
            }
        elif self.config.scheduler.lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.lr_patience,
                gamma=self.config.lr_decay
            )
            scheduler_config = {
                "scheduler": scheduler,
                "frequency": 1
            }
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config
        }