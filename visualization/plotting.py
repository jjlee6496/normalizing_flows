import torch
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn.functional as F

class ImageFlowVisualizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def visualize_reconstructions(self, input_images, n_samples=8):
        """시각화: 원본 이미지와 재구성된 이미지 비교"""
        with torch.no_grad():
            # Forward pass
            z_list, _ = self.model(input_images[:n_samples])
            reconstructed, _ = self.model(z_list, reverse=True)
            
            # 이미지 정규화 및 범위 조정
            input_images = input_images[:n_samples].clamp(0, 1)
            reconstructed = reconstructed.clamp(0, 1)
            
            # 차이 계산 및 시각화를 위한 스케일링
            diff = torch.abs(input_images - reconstructed)
            diff = diff / diff.max()  # [0, 1] 범위로 정규화
            
            # 이미지 그리드 생성
            comparison = torch.cat([
                input_images,
                torch.ones_like(input_images[:, :, :, :3]),  # 흰색 구분선
                reconstructed,
                torch.ones_like(input_images[:, :, :, :3]),  # 흰색 구분선
                diff
            ], dim=3)
            
            grid = make_grid(
                comparison, 
                nrow=1,  # 세로로 쌓기
                normalize=False,  # 이미 정규화했으므로
                padding=5,
                pad_value=1.0  # 흰색 패딩
            )
            
            # matplotlib으로 시각화
            plt.figure(figsize=(15, 5 * n_samples))
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.title('Original | Reconstructed | Difference', pad=20)
            plt.axis('off')
            
            return grid
    
    def slerp(self, z1, z2, alpha):
        """단순한 spherical linear interpolation"""
        # 1. normalize
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        # 2. compute angle
        cos_angle = (z1_norm * z2_norm).sum(dim=1, keepdim=True)
        angle = torch.acos(torch.clamp(cos_angle, -1+1e-6, 1-1e-6))
        # 3. interpolate
        sin_angle = torch.sin(angle)
        t1 = torch.sin((1 - alpha) * angle) / sin_angle
        t2 = torch.sin(alpha * angle) / sin_angle
        return t1 * z1 + t2 * z2

    def _interpolate_latents(self, z1, z2, alpha, method):
        """Helper function for latent interpolation"""
        result = []
        for z1_tensor, z2_tensor in zip(z1, z2):
            if method == 'spherical':
                z1_flat = z1_tensor.view(z1_tensor.size(0), -1)
                z2_flat = z2_tensor.view(z2_tensor.size(0), -1)
                z_interp_flat = self.slerp(z1_flat, z2_flat, alpha)
                z_interp = z_interp_flat.view_as(z1_tensor)
            else:
                z_interp = (1 - alpha) * z1_tensor + alpha * z2_tensor
            result.append(z_interp)
        return result if len(result) > 1 else result[0]
    
    def visualize_interpolations(self, corner_images, n_steps=8, method='spherical'):
        """2D grid interpolation between four corner images"""
        with torch.no_grad():
            # Encode corner images
            z_corners = []
            for img in corner_images:
                z, _ = self.model(img.unsqueeze(0).to(self.device))
                if not isinstance(z, list):
                    z = [z]
                z_corners.append(z)
            
            # Create interpolation weights
            alphas = torch.linspace(0, 1, n_steps).to(self.device)
            beta_grid, alpha_grid = torch.meshgrid(alphas, alphas, indexing='ij')
            
            interpolated_images = []
            for i in range(n_steps):
                row_images = []
                for j in range(n_steps):
                    alpha, beta = alpha_grid[i, j], beta_grid[i, j]
                    
                    # Interpolate horizontally then vertically
                    if method == 'spherical':
                        top_interp = self._interpolate_latents(z_corners[0], z_corners[1], alpha, 'spherical')
                        bottom_interp = self._interpolate_latents(z_corners[2], z_corners[3], alpha, 'spherical')
                        final_interp = self._interpolate_latents(top_interp, bottom_interp, beta, 'spherical')
                    else:
                        top_interp = self._interpolate_latents(z_corners[0], z_corners[1], alpha, 'linear')
                        bottom_interp = self._interpolate_latents(z_corners[2], z_corners[3], alpha, 'linear')
                        final_interp = self._interpolate_latents(top_interp, bottom_interp, beta, 'linear')
                    
                    img_interp, _ = self.model(final_interp, reverse=True)
                    row_images.append(img_interp)
                
                interpolated_images.append(torch.cat(row_images, dim=0))
            
            interpolated_batch = torch.cat(interpolated_images, dim=0)
            return make_grid(interpolated_batch.clamp(0, 1), nrow=n_steps, padding=2)
    
    def visualize_attribute_manipulation(self, input_images, attribute_vectors, alpha_range=(-2, 2), n_steps=8):
        """속성 조작 시각화"""
        with torch.no_grad():
            image = input_images[0].unsqueeze(0)
            z_list, _ = self.model(image)
            
            results = []
            for attr_vec in attribute_vectors:
                attr_images = []
                for alpha in np.linspace(alpha_range[0], alpha_range[1], n_steps):
                    z_new_list = [z + alpha * vec for z, vec in zip(z_list, attr_vec)]
                    img_new, _ = self.model(z_new_list, reverse=True)
                    attr_images.append(img_new)
                    
                attr_grid = make_grid(torch.cat(attr_images).clamp(0, 1), nrow=n_steps)
                results.append(attr_grid)
                
            return results

    def plot_flow(self, input_batch, save_path=None):
        plt.close('all')
        """종합적인 flow 시각화"""
        if save_path:
            save_dir = Path(save_path).parent
            stem = Path(save_path).stem
            
            # 1. 재구성
            recon_grid = self.visualize_reconstructions(input_batch)
            plt.figure(figsize=(15, 5))
            plt.imshow(recon_grid.permute(1, 2, 0).cpu())
            plt.title('Original vs Reconstructed', fontsize=12, pad=10)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_dir / f"{stem}_reconstruction.png", bbox_inches='tight', dpi=300)
            plt.close()
            
            # 2. 선형 보간
            if len(input_batch) >= 4:
                corner_images = input_batch[:4]
                linear_interp = self.visualize_interpolations(corner_images, method='linear')
                plt.figure(figsize=(15, 15))
                plt.imshow(linear_interp.permute(1, 2, 0).cpu())
                plt.title('Linear Interpolation', fontsize=12, pad=10)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(save_dir / f"{stem}_linear.png", bbox_inches='tight', dpi=300)
                plt.close()
                
                # 3. 구면 보간
                spherical_interp = self.visualize_interpolations(corner_images, method='spherical')
                plt.figure(figsize=(15, 15))
                plt.imshow(spherical_interp.permute(1, 2, 0).cpu())
                plt.title('Spherical Interpolation', fontsize=12, pad=10)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(save_dir / f"{stem}_spherical.png", bbox_inches='tight', dpi=300)
                plt.close()
        
        return None