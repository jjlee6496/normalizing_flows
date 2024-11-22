import unittest
import torch
import torch.nn as nn
from models.glow import (
    ActNorm, 
    InvertibleConv1x1, 
    AffineCoupling, 
    squeeze, 
    FlowStep, 
    Glow
)

class TestGlowComponents(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)  # 재현성을 위한 시드 설정
        self.batch_size = 4
        self.channels = 6
        self.height = 16
        self.width = 16
        self.hidden_channels = 12
        self.test_input = torch.randn(
            self.batch_size, 
            self.channels, 
            self.height, 
            self.width
        ) * 0.1  # 입력값의 범위를 제한

    def test_actnorm(self):
        """ActNorm 레이어 테스트"""
        actnorm = ActNorm(self.channels)
        # Forward pass
        output, logdet_fwd = actnorm(self.test_input)
        self.assertEqual(
            output.shape, 
            self.test_input.shape, 
            "Output shape should match input shape"
        )
        
        # Reverse pass
        reversed_output, logdet_rev = actnorm(output, reverse=True)
        self.assertTrue(
            torch.allclose(self.test_input, reversed_output, rtol=1e-4, atol=1e-4),
            f"Max reconstruction error: {torch.max(torch.abs(self.test_input - reversed_output))}"
        )
        # logdet 일관성 체크
        self.assertTrue(
            torch.allclose(logdet_fwd, -logdet_rev, rtol=1e-4),
            f"Logdet inconsistency: {torch.max(torch.abs(logdet_fwd + logdet_rev))}"
        )

    def test_invertible_conv(self):
        """1x1 Invertible Convolution 테스트"""
        conv = InvertibleConv1x1(self.channels)
        # Forward pass
        output, logdet_fwd = conv(self.test_input)
        self.assertEqual(
            output.shape, 
            self.test_input.shape, 
            "Output shape should match input shape"
        )
        
        # Reverse pass
        reversed_output, logdet_rev = conv(output, reverse=True)
        self.assertTrue(
            torch.allclose(self.test_input, reversed_output, rtol=1e-4, atol=1e-4),
            f"Max reconstruction error: {torch.max(torch.abs(self.test_input - reversed_output))}\n"
            f"Mean reconstruction error: {torch.mean(torch.abs(self.test_input - reversed_output))}"
        )
        # logdet 일관성 체크
        self.assertTrue(
            torch.allclose(logdet_fwd, -logdet_rev, rtol=1e-4),
            f"Logdet inconsistency: {torch.max(torch.abs(logdet_fwd + logdet_rev))}"
        )

    def test_affine_coupling(self):
        """Affine Coupling Layer 테스트"""
        coupling = AffineCoupling(self.channels, self.hidden_channels)
        # Forward pass
        output, logdet_fwd = coupling(self.test_input)
        self.assertEqual(
            output.shape, 
            self.test_input.shape, 
            "Output shape should match input shape"
        )
        
        # Reverse pass
        reversed_output, logdet_rev = coupling(output, reverse=True)
        self.assertTrue(
            torch.allclose(self.test_input, reversed_output, rtol=1e-4, atol=1e-4),
            f"Max reconstruction error: {torch.max(torch.abs(self.test_input - reversed_output))}"
        )
        # logdet 일관성 체크
        self.assertTrue(
            torch.allclose(logdet_fwd, -logdet_rev, rtol=1e-4),
            f"Logdet inconsistency: {torch.max(torch.abs(logdet_fwd + logdet_rev))}"
        )

    def test_squeeze(self):
        """Squeeze 연산 테스트"""
        # Forward squeeze
        squeezed = squeeze(self.test_input)
        expected_shape = (
            self.batch_size,
            self.channels * 4,
            self.height // 2,
            self.width // 2
        )
        self.assertEqual(
            squeezed.shape, 
            expected_shape, 
            "Squeezed output shape is incorrect"
        )
        
        # Reverse squeeze
        unsqueezed = squeeze(squeezed, reverse=True)
        self.assertEqual(
            unsqueezed.shape,
            self.test_input.shape,
            "Unsqueezed output shape should match input shape"
        )
        self.assertTrue(
            torch.allclose(self.test_input, unsqueezed, rtol=1e-4, atol=1e-4),
            f"Max reconstruction error: {torch.max(torch.abs(self.test_input - unsqueezed))}"
        )

    def test_flow_step(self):
        """Single Flow Step 테스트"""
        flow_step = FlowStep(self.channels, self.hidden_channels)
        # Forward pass
        output, logdet_fwd = flow_step(self.test_input)
        self.assertEqual(
            output.shape, 
            self.test_input.shape, 
            "Output shape should match input shape"
        )
        
        # Reverse pass
        reversed_output, logdet_rev = flow_step(output, reverse=True)
        self.assertTrue(
            torch.allclose(self.test_input, reversed_output, rtol=1e-4, atol=1e-4),
            f"Max reconstruction error: {torch.max(torch.abs(self.test_input - reversed_output))}\n"
            f"Mean reconstruction error: {torch.mean(torch.abs(self.test_input - reversed_output))}"
        )
        # logdet 일관성 체크
        self.assertTrue(
            torch.allclose(logdet_fwd, -logdet_rev, rtol=1e-4),
            f"Logdet inconsistency: {torch.max(torch.abs(logdet_fwd + logdet_rev))}"
        )

class TestGlowModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.in_channels = 3
        self.hidden_channels = 12
        self.K = 2  # Number of flow steps
        self.L = 3  # Number of scales
        self.height = 32
        self.width = 32
        
        self.model = Glow(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            K=self.K,
            L=self.L
        )
        
        self.test_input = torch.randn(
            self.batch_size, 
            self.in_channels, 
            self.height, 
            self.width
        ) * 0.1

    def test_glow_forward_reverse(self):
        """전체 Glow 모델의 forward/reverse 테스트"""
        # Forward pass
        z, logdet_fwd = self.model(self.test_input)
        
        # Check that we get the expected number of outputs
        self.assertEqual(len(z), self.L, "Number of outputs should match L")
        
        # Reverse pass
        reversed_output, logdet_rev = self.model(z, reverse=True)
        
        # Check shapes
        self.assertEqual(
            reversed_output.shape, 
            self.test_input.shape,
            "Reconstructed output shape should match input shape"
        )
        
        # Check reversibility
        self.assertTrue(
            torch.allclose(self.test_input, reversed_output, rtol=1e-4, atol=1e-4),
            f"Max reconstruction error: {torch.max(torch.abs(self.test_input - reversed_output))}\n"
            f"Mean reconstruction error: {torch.mean(torch.abs(self.test_input - reversed_output))}"
        )

    def test_multi_scale_output_shapes(self):
        """Multi-scale 아키텍처의 출력 shape 테스트"""
        z, _ = self.model(self.test_input)
        
        # Calculate expected shapes for each scale
        current_channels = self.in_channels
        current_size = self.height
        
        for i in range(self.L):
            current_channels *= 4
            current_size //= 2
            
            if i < self.L - 1:
                expected_shape = (
                    self.batch_size, 
                    current_channels // 2, 
                    current_size, 
                    current_size
                )
                current_channels //= 2
            else:
                expected_shape = (
                    self.batch_size, 
                    current_channels, 
                    current_size, 
                    current_size
                )
            
            self.assertEqual(
                z[i].shape, 
                expected_shape,
                f"Output shape at level {i} is incorrect"
            )

if __name__ == '__main__':
    unittest.main()