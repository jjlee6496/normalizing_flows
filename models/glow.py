import torch
import torch.nn as nn
import torch.nn.functional as F

class ActNorm(nn.Module):
    """Activation Normalization Layer
    입력 데이터를 채널별로 정규화하는 레이어
    """
    def __init__(self, channels):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.register_buffer('initialized', torch.tensor(0))
        
    def initialize(self, x):
        """데이터 기반 초기화
        Input shape: [B, C, H, W]
        """
        with torch.no_grad():
            # 각 채널별 평균과 표준편차 계산
            mean = x.mean(dim=(0, 2, 3), keepdim=True)  # [1, C, 1, 1]
            std = x.std(dim=(0, 2, 3), keepdim=True)    # [1, C, 1, 1]
            
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))
            
    def forward(self, x, reverse=False):
        """
        Input shape: [B, C, H, W]
        Output shape: [B, C, H, W] (차원 변화 없음)
        """
        if x.device != self.loc.device:
            self.to(x.device)
            
        b, _, h, w = x.shape
        if self.initialized.item() == 0 and not reverse:
            self.initialize(x)
            self.initialized.fill_(1)

        if reverse:
            out = (x - self.loc) / self.scale
        else:
            out = x * self.scale + self.loc

        logdet = torch.sum(torch.log(torch.abs(self.scale))) * h * w
        if reverse:
            logdet = -logdet

        return out, logdet.repeat(b)

class InvertibleConv1x1(nn.Module):
    """1x1 Invertible Convolution
    채널 간의 정보를 섞는 가역적 변환
    """
    def __init__(self, channels):
        super().__init__()
        # orthogonal matrix initialization
        w_init = torch.randn(channels, channels)
        w_init = torch.linalg.qr(w_init)[0].contiguous()
        self.weight = nn.Parameter(w_init)
        
    def forward(self, x, reverse=False):
        """
        Input shape: [B, C, H, W]
        Output shape: [B, C, H, W]
        """
        if x.device != self.weight.device:
            self.to(x.device)
            
        b, c, h, w = x.shape
        weight = self.weight
        
        if reverse:
            weight = torch.linalg.solve(
                weight.to(dtype=torch.float64), 
                torch.eye(c, dtype=torch.float64, device=x.device)
            ).to(dtype=torch.float32)
            logdet = -h * w * torch.slogdet(self.weight)[1]
        else:
            logdet = h * w * torch.slogdet(self.weight)[1]
            
        weight = weight.view(c, c, 1, 1)
        z = F.conv2d(x, weight)
        return z, logdet.repeat(b)

class AffineCoupling(nn.Module):
    """Affine Coupling Layer
    입력을 절반으로 나누어 한쪽으로 다른쪽을 변환
    """
    def __init__(self, channels, hidden_channels):
        super().__init__()
        # 변환 함수 (입력 채널의 절반 -> 전체 채널)
        self.net = nn.Sequential(
            nn.Conv2d(channels//2, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 3, padding=1)
        )
        
    def forward(self, x, reverse=False):
        """
        Input shape: [B, C, H, W]
        Output shape: [B, C, H, W] (차원 변화 없음)
        """
        if x.device != next(self.parameters()).device:
            self.to(x.device)
            
        x1, x2 = x.chunk(2, dim=1)  # 채널 차원으로 반으로 나눔
        h = self.net(x1)  # [B, C, H, W]
        shift, scale = h.chunk(2, dim=1)  # 각각 [B, C/2, H, W]
        scale = torch.sigmoid(scale + 2.)  # scale 값을 항상 양수로 만듦
        
        log_scale = torch.log(scale)
        logdet = torch.sum(log_scale, dim=[1, 2, 3])
        
        if reverse:
            x2 = (x2 - shift) / scale
            return torch.cat([x1, x2], dim=1), -logdet
        else:
            x2 = x2 * scale + shift
            return torch.cat([x1, x2], dim=1), logdet


def squeeze(x, reverse=False):
    """Space-to-depth / Depth-to-space operation
    Input shape: [B, C, H, W]
    Output shape: [B, C*4, H/2, W/2] (정방향)
    또는 [B, C/4, H*2, W*2] (역방향)
    """
    b, c, h, w = x.shape
    if not reverse:
        x = x.view(b, c, h//2, 2, w//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c*4, h//2, w//2)
    else:
        x = x.view(b, c//4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c//4, h*2, w*2)
    return x

class FlowStep(nn.Module):
    """Single Flow Step combining ActNorm, 1x1 conv, and Coupling"""
    def __init__(self, channels, hidden_channels):
        super().__init__()
        self.actnorm = ActNorm(channels)
        self.conv1x1 = InvertibleConv1x1(channels)
        self.coupling = AffineCoupling(channels, hidden_channels)
    
    def forward(self, x, reverse=False):
        """각 step에서 차원 변화 없음
        Input/Output shape: [B, C, H, W]
        Returns: (output, logdet)
        """
        if not reverse:
            # forward order: actnorm -> conv1x1 -> coupling
            x, logdet1 = self.actnorm(x)
            x, logdet2 = self.conv1x1(x)
            x, logdet3 = self.coupling(x)
            return x, logdet1 + logdet2 + logdet3
        else:
            # reverse order: coupling -> conv1x1 -> actnorm
            x, logdet3 = self.coupling(x, reverse=True)
            x, logdet2 = self.conv1x1(x, reverse=True)
            x, logdet1 = self.actnorm(x, reverse=True)
            # Note: logdet signs are already handled in individual components
            return x, logdet1 + logdet2 + logdet3

class Glow(nn.Module):
    """Complete Glow model with multi-scale architecture"""
    def __init__(self, in_channels, hidden_channels, K, L):
        """
        Args:
            in_channels: 입력 채널 수 (e.g., RGB면 3)
            hidden_channels: coupling network의 hidden dimension
            K: 각 scale에서의 flow step 수
            L: scale의 수
        """
        super().__init__()
        self.flows = nn.ModuleList()
        self.K = K
        self.L = L
        self.in_channels = in_channels
        
        c = in_channels
        for i in range(L):
            c *= 4  # squeeze로 인한 채널 수 증가
            for _ in range(K):
                self.flows.append(FlowStep(c, hidden_channels))
            if i < L-1:
                c = c // 2  # split으로 인한 채널 수 감소
    
    @property
    def base_dist(self):
        """Return base distribution"""
        return lambda x: torch.distributions.Normal(
            loc=torch.zeros_like(x),
            scale=torch.ones_like(x)
        )
    
    def forward(self, x, reverse=False):
        """
        Forward:
            Input: [B, C, H, W]
            Output: (z_list, logdet)
        Reverse:
            Input: List of tensors
            Output: ([B, C, H, W], logdet)
        """
        if not reverse:
            z_list = []
            logdet = 0
            
            for i in range(self.L):
                x = squeeze(x)
                
                for k in range(self.K):
                    x, det = self.flows[i * self.K + k](x)
                    logdet = logdet + det
                
                if i < self.L-1:
                    x, z = x.chunk(2, dim=1)
                    z_list.append(z)
            
            z_list.append(x)
            return z_list, logdet
        else:
            if not isinstance(x, list):
                raise ValueError("Input to reverse mode must be a list of tensors")
            
            z_list = x
            logdet = 0
            x = z_list[-1]
            
            for i in reversed(range(self.L)):
                if i < self.L-1:
                    x = torch.cat([x, z_list[i]], dim=1)
                
                for k in reversed(range(self.K)):
                    x, det = self.flows[i * self.K + k](x, reverse=True)
                    logdet = logdet + det
                
                x = squeeze(x, reverse=True)
            
            return x, logdet
    
    def sample(self, batch_size, device=None):
        """Generate samples from the model
        Args:
            batch_size: Number of samples to generate
            device: Device to generate samples on
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Generate list of z's for each scale
        z_list = []
        c = self.in_channels
        h, w = 64, 64  # 초기 이미지 크기
        
        for i in range(self.L):
            c *= 4  # squeeze로 인한 채널 수 증가
            if i < self.L - 1:
                # Split operation
                current_c = c // 2
                c = c // 2
            else:
                current_c = c
                
            z = torch.randn(batch_size, current_c, h // (2 ** (i + 1)), 
                           w // (2 ** (i + 1)), device=device)
            z_list.append(z)
        
        # Generate samples using reverse flow
        return self.forward(z_list, reverse=True)