from torch import Tensor
from typing import List, Optional, Sequence, Union
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA

class MyCelebA(CelebA):
    """
    체크섬 생략
    """
    def _check_integrity(self) -> bool:
        return True
    

class CelebADataset(LightningDataModule):
    """
    pytorch lightning data module

    Args:
        data_dir: 데이터셋 루트 디렉토리
        train_batch_size: train 배치사이즈
        val_batch_size: val 배치 사이즈
        patch_size: 원본이미지에서 크롭할 크기
        num_workers: parallel workers to create to load data items(Dataloader 문서 참고)
        pin_memory: whether prepared items should be loaded into pinned memory
                    or not. this can improve perfermance on GPUs.
    """
    def __init__(
        self,
        data_dir: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                             transforms.CenterCrop(148),
                             transforms.Resize(self.patch_size),
                             transforms.ToTensor(),])
        
        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                             transforms.CenterCrop(148),
                             transforms.Resize(self.patch_size),
                             transforms.ToTensor(),])
        
        self.train_dataset = MyCelebA(
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )
        
        self.val_dataset = MyCelebA(
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )