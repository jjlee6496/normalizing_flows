# scripts/evaluate.py
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

from configs import RealNVPConfig, MAFConfig, GlowConfig
from models import RealNVP, MAF, Glow
from visualization.plotting import ImageFlowVisualizer
from utils.setup import setup_experiment
from utils.metrics import calculate_ssim, InceptionStatistics, calculate_fid
from dataset import MyCelebA

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["realnvp", "maf", "glow"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_path", type=str, default="data/celeba")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples for evaluation")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()

def load_model_and_config(args):
    """Load model and config from checkpoint"""
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Create config based on model type
        if args.model == "realnvp":
            config = RealNVPConfig()
        elif args.model == "maf":
            config = MAFConfig()
        else:
            config = GlowConfig()
    
    # Update config with evaluation settings
    config.data_dir = args.data_path
    config.output_dir = args.output_dir
    
    # Create model
    if config.model_type == "realnvp":
        model = RealNVP(
            input_dims=(3, 64, 64),
            hidden_dims=config.hidden_dims,
            n_blocks=config.n_blocks,
            mask_type=config.mask_type,
            coupling_type=config.coupling_type
        )
    elif config.model_type == "maf":
        model = MAF(
            input_size=3*64*64,
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
    
    # Load model weights
    model.load_state_dict(checkpoint['state_dict'])
    return model, config

def evaluate_reconstruction(model, dataloader, device, save_dir):
    """Evaluate reconstruction quality"""
    model.eval()
    ssim_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating reconstruction"):
            images = batch[0].to(device)
            
            # Reconstruction
            z, _ = model(images)
            reconstructed, _ = model(z, reverse=True)
            
            # Compute SSIM
            ssim = calculate_ssim(images, reconstructed)
            ssim_scores.append(ssim.item())
            
            # Save some reconstructions
            if len(ssim_scores) == 1:  # Save first batch
                comparison = torch.cat([images[:8], reconstructed[:8]])
                save_image(comparison, f"{save_dir}/reconstructions.png", nrow=8)
    
    return np.mean(ssim_scores)

def evaluate_sampling(model, n_samples, device, save_dir):
    """Evaluate sampling quality"""
    model.eval()
    inception = InceptionStatistics(device)
    
    with torch.no_grad():
        # Generate samples
        samples = []
        for _ in tqdm(range(0, n_samples, 100), desc="Generating samples"):
            batch_size = min(100, n_samples - len(samples)*100)
            sample = model.sample(batch_size)
            samples.append(sample.cpu())
        
        samples = torch.cat(samples)
        
        # Save sample grid
        sample_grid = make_grid(samples[:64], nrow=8)
        save_image(sample_grid, f"{save_dir}/samples.png")
        
        # Compute inception features
        inception_features = inception.get_activations(samples)
        
    return inception_features

def evaluate_interpolation(model, dataloader, device, save_dir):
    """Evaluate latent space interpolation"""
    model.eval()
    visualizer = ImageFlowVisualizer(model, device)
    
    with torch.no_grad():
        # Get pair of images
        images = next(iter(dataloader))[0][:2].to(device)
        
        # Generate both linear and spherical interpolations
        linear_interp = visualizer.visualize_interpolations(images, method='linear')
        spherical_interp = visualizer.visualize_interpolations(images, method='spherical')
        
        # Save results
        save_image(linear_interp, f"{save_dir}/linear_interpolation.png")
        save_image(spherical_interp, f"{save_dir}/spherical_interpolation.png")

def evaluate_model(model, config, dataloader, device):
    """Comprehensive model evaluation"""
    # Create output directory
    save_dir = Path(config.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Reconstruction quality
    ssim_score = evaluate_reconstruction(model, dataloader, device, save_dir)
    
    # 2. Sampling quality
    sample_features = evaluate_sampling(model, config.n_samples, device, save_dir)
    
    # 3. Interpolation
    evaluate_interpolation(model, dataloader, device, save_dir)
    
    # 4. Compute FID score
    with torch.no_grad():
        real_features = []
        for batch in tqdm(dataloader, desc="Computing real features"):
            images = batch[0].to(device)
            inception = InceptionStatistics(device)
            features = inception.get_activations(images)
            real_features.append(features.cpu())
        real_features = torch.cat(real_features)
        
        fid_score = calculate_fid(real_features, sample_features)
    
    # Save metrics
    metrics = {
        'ssim_score': ssim_score,
        'fid_score': fid_score,
    }
    
    # Save metrics to file
    with open(f"{save_dir}/metrics.txt", 'w') as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value}\n")
    
    return metrics

def main():
    args = parse_args()
    
    # Load model and config
    model, config = load_model_and_config(args)
    
    # Setup
    device = setup_experiment(config)
    model = model.to(device)
    model.eval()
    
    # Create data loader
    data_module = MyCelebA(
        data_path=config.data_dir,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )
    data_module.setup()
    
    # Evaluate
    metrics = evaluate_model(model, config, data_module.val_dataloader(), device)
    
    print("\nEvaluation Results:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

if __name__ == "__main__":
    main()