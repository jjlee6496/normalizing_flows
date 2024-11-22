import unittest
import warnings
import torch
import numpy as np
from utils.metrics import (
    calculate_ssim,
    create_window,
    InceptionStatistics,
    calculate_fid,
    calculate_inception_score
)
from torchvision.models import Inception_V3_Weights

class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Suppress warnings for the entire test suite
        warnings.filterwarnings("ignore", category=UserWarning)
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.batch_size = 32
        self.channels = 3
        self.height = 64
        self.width = 64

    def test_create_window(self):
        """Test gaussian window creation"""
        window_size = 11
        window = create_window(window_size)
        
        self.assertEqual(window.shape, (1, 1, window_size, window_size))
        self.assertTrue(torch.abs(window.sum() - 1.0) < 1e-6)
        self.assertTrue(torch.allclose(window, window.flip(-1)))
        self.assertTrue(torch.allclose(window, window.flip(-2)))
    
    def test_ssim(self):
        """Test SSIM calculation with CelebA-sized images"""
        img1 = torch.rand(self.batch_size, self.channels, self.height, self.width).to(self.device)
        img2 = img1.clone()
        noisy_img = img1 + 0.1 * torch.randn_like(img1)
        
        ssim_same = calculate_ssim(img1, img2)
        ssim_noisy = calculate_ssim(img1, noisy_img)
        
        self.assertTrue(torch.abs(ssim_same - 1.0) < 1e-6)
        self.assertTrue(ssim_noisy < ssim_same)
        self.assertTrue(ssim_noisy > 0)
    
    def test_inception_statistics(self):
        """Test Inception feature extraction with CelebA-sized images"""
        inception = InceptionStatistics(self.device)
        
        images = torch.rand(self.batch_size, self.channels, self.height, self.width).to(self.device)
        activations = inception.get_activations(images)
        
        self.assertEqual(activations.shape[0], self.batch_size)
        self.assertEqual(str(activations.device), str(self.device))
    
    def test_fid(self):
        """Test FID calculation with NaN/inf validation."""
        n_samples = 1000
        n_features = 2048
        
        rng = np.random.RandomState(42)
        real_acts = rng.randn(n_samples, n_features)
        fake_acts = rng.randn(n_samples, n_features) * 1.1 + 0.5
        same_acts = real_acts.copy()

        # Check for NaN or inf values
        self.assertFalse(np.isnan(real_acts).any(), "real_acts contains NaN values.")
        self.assertFalse(np.isinf(real_acts).any(), "real_acts contains inf values.")
        self.assertFalse(np.isnan(fake_acts).any(), "fake_acts contains NaN values.")
        self.assertFalse(np.isinf(fake_acts).any(), "fake_acts contains inf values.")

        fid_diff = calculate_fid(real_acts, fake_acts)
        fid_same = calculate_fid(real_acts, same_acts)
        
        self.assertTrue(fid_same < fid_diff)
        self.assertTrue(np.abs(fid_same) < 1e-4)
    
    def test_inception_score(self):
        """Test Inception Score calculation"""
        n_samples = 1000
        n_classes = 1000
        
        uniform_acts = torch.ones(n_samples, n_classes) / n_classes
        peaked_acts = torch.zeros(n_samples, n_classes)
        peaked_acts[range(n_samples), torch.randint(0, n_classes, (n_samples,))] = 10.0
        
        is_uniform_mean, is_uniform_std = calculate_inception_score(uniform_acts)
        is_peaked_mean, is_peaked_std = calculate_inception_score(peaked_acts)
        
        self.assertTrue(is_peaked_mean > is_uniform_mean)
        self.assertTrue(is_uniform_mean > 0)
        self.assertTrue(is_peaked_std >= 0)
    
    def test_input_validation(self):
        """Test input validation and error handling"""
        # Device를 맞춰서 테스트
        img1 = torch.rand(self.batch_size, self.channels, self.height, self.width).to(self.device)
        
        # 각각의 에러 케이스를 개별적으로 테스트
        # 1. 다른 크기의 이미지
        img2 = torch.rand(self.batch_size, self.channels, self.height * 2, self.width * 2).to(self.device)
        self.assertRaises(RuntimeError, calculate_ssim, img1, img2)
        
        # 2. 잘못된 채널 수
        wrong_channels = torch.rand(self.batch_size, 1, self.height, self.width).to(self.device)
        self.assertRaises(RuntimeError, calculate_ssim, img1, wrong_channels)
        
        # 3. 잘못된 배치 크기
        wrong_batch = torch.rand(self.batch_size * 2, self.channels, self.height, self.width).to(self.device)
        self.assertRaises(RuntimeError, calculate_ssim, img1, wrong_batch)
        
        # 4. 부동소수점이 아닌 텐서
        int_tensor = torch.randint(0, 255, (self.batch_size, self.channels, self.height, self.width), 
                                 dtype=torch.uint8).to(self.device)
        self.assertRaises(RuntimeError, calculate_ssim, img1, int_tensor)
        
    def test_input_device_consistency(self):
        """Test device consistency requirements"""
        img1 = torch.rand(self.batch_size, self.channels, self.height, self.width).to(self.device)
        img2 = torch.rand(self.batch_size, self.channels, self.height, self.width)  # CPU tensor
        
        self.assertRaises(RuntimeError, calculate_ssim, img1, img2)

if __name__ == '__main__':
    unittest.main()