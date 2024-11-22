import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg

# SSIM 계산 함수
def calculate_ssim(img1, img2, window_size=11):
    """Calculate SSIM for RGB images
    Args:
        img1: [B, C, H, W] tensor
        img2: [B, C, H, W] tensor
        window_size: size of the gaussian window
    """
    # Create gaussian window
    window = create_window(window_size).to(img1.device)
    
    # Calculate SSIM for each channel and average
    ssim_value = 0
    for c in range(img1.shape[1]):  # For each channel
        ssim_value += _ssim(img1[:, c:c+1, :, :], 
                           img2[:, c:c+1, :, :], 
                           window, 
                           window_size)
    
    return ssim_value / img1.shape[1]  # Average over channels

def _ssim(img1, img2, window, window_size):
    """Calculate SSIM for single channel images"""
    mu1 = F.conv2d(img1, window, padding=window_size // 2)
    mu2 = F.conv2d(img2, window, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()

def create_window(window_size, sigma=1.5):
    """Create a gaussian window"""
    x = torch.arange(window_size)
    x = x - (window_size - 1) / 2
    if len(x.shape) == 1:
        x = x[None, :]
    gauss = torch.exp(-(x ** 2 + x.T ** 2) / (2 * sigma ** 2))
    return gauss.unsqueeze(0).unsqueeze(0) / gauss.sum()

class InceptionStatistics:
    def __init__(self, device):
        self.device = 'cuda:0'
        # Inception 모델 초기화 및 device로 이동
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.model.to(device).eval()
        
        # Gradient 계산 비활성화
        for param in self.model.parameters():
            param.requires_grad = False
    
    def get_activations(self, images):
        """Get activations from the penultimate layer of the Inception model."""
        with torch.no_grad():
            try:
                images = images.to(self.device, dtype=torch.float32)
                
                # 이미지 크기 조정이 필요한 경우
                if images.size(1) != 3 or images.size(2) < 299 or images.size(3) < 299:
                    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
                
                # 새로운 autocast API 사용
                with torch.amp.autocast('cuda', enabled=False):
                    outputs = self.model(images)
                    
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                return outputs
                
            except RuntimeError as e:
                print(f"Runtime Error in get_activations: {e}")
                print(f"Input tensor info - dtype: {images.dtype}, device: {images.device}, shape: {images.shape}")
                raise

def calculate_fid(real_acts, fake_acts):
    """Calculate Frechet Inception Distance (FID) between real and fake activations.
    
    Parameters:
        real_acts: Tensor or array of real activations
        fake_acts: Tensor or array of fake/generated activations
        
    Returns:
        float: Calculated FID value, or inf if calculation fails
    """
    # Convert to numpy if needed
    if isinstance(real_acts, torch.Tensor):
        real_acts = real_acts.float().cpu().numpy()
    if isinstance(fake_acts, torch.Tensor):
        fake_acts = fake_acts.float().cpu().numpy()
    
    # Initial check for NaN/inf values
    if np.any(np.isnan(real_acts)) or np.any(np.isinf(real_acts)) or \
       np.any(np.isnan(fake_acts)) or np.any(np.isinf(fake_acts)):
        print("Warning: Input arrays contain NaN or inf values. Attempting to clean...")
        real_acts = np.nan_to_num(real_acts, nan=0.0, posinf=0.0, neginf=0.0)
        fake_acts = np.nan_to_num(fake_acts, nan=0.0, posinf=0.0, neginf=0.0)
    
    try:
        # Calculate means and covariances
        mu1 = np.mean(real_acts, axis=0)
        mu2 = np.mean(fake_acts, axis=0)
        
        # Add small epsilon to diagonal for numerical stability
        epsilon = 1e-6
        sigma1 = np.cov(real_acts, rowvar=False) + np.eye(real_acts.shape[1]) * epsilon
        sigma2 = np.cov(fake_acts, rowvar=False) + np.eye(fake_acts.shape[1]) * epsilon
        
        # Verify computed statistics
        if np.any(np.isnan(sigma1)) or np.any(np.isnan(sigma2)) or \
           np.any(np.isinf(sigma1)) or np.any(np.isinf(sigma2)):
            print("Warning: Covariance matrices contain NaN or inf values.")
            return float('inf')
        
        # Calculate sqrt(sigma1 @ sigma2)
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        
        # Check if result is complex
        if np.iscomplexobj(covmean):
            if not np.allclose(covmean.imag, 0, rtol=1e-3):
                print("Warning: Imaginary component is non-negligible")
            covmean = covmean.real
        
        # Calculate FID
        diff = mu1 - mu2
        fid = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        
        # Final validation
        if not np.isfinite(fid):
            print("Warning: Final FID value is NaN or inf")
            return float('inf')
        
        return float(fid)
        
    except Exception as e:
        print(f"Error in FID calculation: {str(e)}")
        return float('inf')

def calculate_inception_score(activations, num_splits=10):
    """Calculate Inception Score for a set of activations."""
    # Convert to torch tensor and ensure float32
    if isinstance(activations, np.ndarray):
        activations = torch.from_numpy(activations)
    activations = activations.float()
    
    scores = []
    for i in range(num_splits):
        part = activations[i * (len(activations) // num_splits): (i + 1) * (len(activations) // num_splits), :]
        p_y = F.softmax(part, dim=1)
        p_y_mean = p_y.mean(dim=0, keepdim=True)
        kl_div = p_y * (p_y.log() - p_y_mean.log())
        kl_div = kl_div.sum(dim=1)
        scores.append(torch.exp(kl_div.mean()).item())
    
    return float(np.mean(scores)), float(np.std(scores))