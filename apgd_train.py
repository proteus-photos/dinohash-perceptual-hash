import argparse
import os
from typing import Optional, List, Dict, Any

from PIL import Image
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from hashes.dinohash import DINOHash, preprocess
from apgd_attack import APGDAttack, criterion_loss
from utils import AverageMeter


# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


class ImageDataset(Dataset):
    """Dataset class for loading and preprocessing images."""
    
    def __init__(self, image_files: List[str]):
        self.image_files = image_files

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return preprocess(Image.open(self.image_files[idx]))


class DINOHashDataModule(L.LightningDataModule):    
    def __init__(
        self, 
        image_dir: str, 
        batch_size: int = 200, 
        split_ratio: float = 0.999,
        num_workers: int = 11
    ):
        super().__init__()
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.num_workers = num_workers
        self.save_hyperparameters()
        
    def setup(self, stage: Optional[str] = None) -> None:
        image_files = [
            os.path.join(self.image_dir, f) 
            for f in os.listdir(self.image_dir) 
            if os.path.isfile(os.path.join(self.image_dir, f))
        ]
        image_files.sort()
        image_files = image_files[:1_800_000]  # Keep original limitation
        
        dataset = ImageDataset(image_files)
        
        if stage == "fit" or stage is None:
            train_size = int(self.split_ratio * len(dataset))
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                dataset, [train_size, val_size]
            )
            
        self.complete_dataset = dataset
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
        
    def complete_dataloader(self) -> DataLoader:
        return DataLoader(
            self.complete_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )


class AdversarialDINOHashModule(L.LightningModule):    
    def __init__(
        self,
        model_name: str = "vits14_reg",
        n_bits: int = 96,
        epsilon: float = 8/255,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup: int = 1400,
        max_steps: int = 20000,
        n_iter: int = 20,
        n_iter_range: int = 0,
        clean_weight: float = 500,
        val_freq: int = 500,
        l2_sp: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.clean_dinohash = DINOHash(
            model=model_name, 
            pca_dims=n_bits, 
            prod_mode=False
        )
        self.adversarial_dinohash = DINOHash(
            model=model_name, 
            pca_dims=n_bits, 
            prod_mode=False
        )
        
        for param in self.clean_dinohash.dinov2.parameters():
            param.requires_grad = False
        self.clean_dinohash.dinov2.eval()
        
        for param in self.adversarial_dinohash.dinov2.parameters():
            param.requires_grad = True
            
        self.apgd = APGDAttack(
            dinohash=self.adversarial_dinohash, 
            eps=epsilon
        )
        
        self.automatic_optimization = False
        
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = AdamW(
            self.adversarial_dinohash.dinov2.parameters(),
            lr=self.hparams.lr,
            weight_decay=0 if self.hparams.l2_sp else self.hparams.weight_decay,
            betas=(0.9, 0.95)
        )
        
        def lr_lambda(step: int) -> float:
            if step < self.hparams.warmup:
                return step / self.hparams.warmup
            else:
                progress = (step - self.hparams.warmup) / (self.hparams.max_steps - self.hparams.warmup)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        
        logits = self.clean_dinohash.hash(batch, differentiable=False, logits=True).float()
        
        # sample base n_iter, then double after half of max_steps
        n_iter = np.random.randint(
            self.hparams.n_iter - self.hparams.n_iter_range,
            self.hparams.n_iter + self.hparams.n_iter_range + 1
        )
        # if self.global_step >= (self.hparams.max_steps // 2):
        #     n_iter *= 2

        eps = np.random.uniform(0.5, 1.5) * self.hparams.epsilon
        
        logits = logits.to(self.device)
        batch = batch.to(self.device)
        
        adv_images, _ = self.apgd.attack_single_run(
            batch, logits, n_iter, log=False, eps=eps
        )

        
        if self.hparams.l2_sp:
            reg_loss = l2_sp(
                self.adversarial_dinohash.dinov2, 
                self.clean_dinohash.dinov2, 
                alpha=self.hparams.weight_decay
            )
        else:
            reg_loss = torch.tensor(0.0, device=self.device)

        self.adversarial_dinohash.dinov2.train()
        
        adv_hashes, adv_loss = criterion_loss(
            adv_images, logits, self.adversarial_dinohash.hash, loss="target bce"
        )
        adv_loss = adv_loss.mean()
        self.manual_backward(adv_loss + reg_loss)
        
        clean_loss = torch.tensor(0.0, device=self.device)
        if self.hparams.clean_weight > 0:
            clean_hashes, clean_loss = criterion_loss(
                batch, logits, self.adversarial_dinohash.hash, loss="target bce"
            )
            clean_loss = self.hparams.clean_weight * clean_loss.mean()
            self.manual_backward(clean_loss)
        
        total_loss = adv_loss + clean_loss

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        hashes = (logits >= 0).float()
        accuracy = (adv_hashes - hashes).abs().mean()
        
        self.log('train/total_loss', total_loss, on_step=True, prog_bar=True)
        self.log('train/adv_loss', adv_loss, on_step=True)
        self.log('train/clean_loss', clean_loss / max(self.hparams.clean_weight, 1), on_step=True)
        self.log('train/accuracy', accuracy * 100, on_step=True, prog_bar=True)
        self.log('train/lr', optimizer.param_groups[0]['lr'], on_step=True)
        
        return total_loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        self.adversarial_dinohash.dinov2.eval()
        
        with torch.no_grad():
            logits = self.clean_dinohash.hash(batch, differentiable=False, logits=True).float()
            hashes = (logits >= 0).float()

            adv_images, _ = self.apgd.attack_single_run(
                batch, logits, n_iter=50, eps=self.hparams.epsilon
            )
            
            adv_hashes = self.adversarial_dinohash.hash(adv_images).float()
            attack_accuracy = (adv_hashes - hashes).abs().mean() * 100
            
            clean_hashes = self.adversarial_dinohash.hash(batch).float()
            clean_accuracy = (clean_hashes - hashes).abs().mean() * 100
        
        # Log the monitored metric
        self.log('val/attack_strength', attack_accuracy, on_epoch=True, prog_bar=True)
        self.log('val/clean_error', clean_accuracy, on_epoch=True, prog_bar=True)

        return {
            'val/attack_strength': attack_accuracy,
            'val/clean_error': clean_accuracy,
            'batch_size': len(batch)
        }

def l2_sp(model1, model2, alpha):
    reg_loss = 0
    params_index = -1
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        params_index += 1
        reg_loss += 0.5* alpha * (param1 - param2).norm(2)**2

    return reg_loss

def main():
    global args

    parser = argparse.ArgumentParser(description='Adversarial neural collision attack with Lightning')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for processing images')
    parser.add_argument('--image_dir', type=str, default='./diffusion_data', help='Directory containing images')
    parser.add_argument('--n_iter', type=int, default=20, help='Average number of iterations')
    parser.add_argument('--n_iter_range', type=int, default=0, help='Maximum number of iterations')
    parser.add_argument('--epsilon', type=float, default=8/255, help='Maximum perturbation (L∞ norm bound)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--warmup', type=int, default=1400, help='Number of warmup steps')
    parser.add_argument('--steps', type=int, default=20000, help='Number of steps')
    parser.add_argument('--start_step', type=int, default=0, help='Starting step')
    parser.add_argument('--clean_weight', type=float, default=500, help='Weight of clean loss')
    parser.add_argument('--val_freq', type=int, default=500, help='Validation frequency')
    parser.add_argument('--resume_path', type=str, default=None, help='Resume path')
    parser.add_argument('--n_bits', type=int, default=96, help='Number of PCA components for DINOHash')
    parser.add_argument('--model_name', type=str, default="vits14_reg", help='Model backbone for DINOv2')
    parser.add_argument('--l2_sp', type=bool, default=True, help='L2 similarity penalty between clean and adversarial DINOHash')
    
    parser.add_argument('--project_name', type=str, default='adversarial-dinohash', help='Wandb project name')
    parser.add_argument('--offline', action='store_true', help='Run wandb in offline mode')
    
    args = parser.parse_args()
    torch.set_float32_matmul_precision('medium')
    
    datamodule = DINOHashDataModule(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        split_ratio=0.999,
        num_workers=11
    )
    
    model = AdversarialDINOHashModule(
        model_name=args.model_name,
        n_bits=args.n_bits,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup=args.warmup,
        max_steps=args.steps,
        epsilon=args.epsilon,
        n_iter=args.n_iter,
        n_iter_range=args.n_iter_range,
        clean_weight=args.clean_weight,
        val_freq=args.val_freq
    )
    
    if args.resume_path:
        checkpoint = torch.load(args.resume_path, weights_only=False)
        model.adversarial_dinohash.dinov2.load_state_dict(checkpoint["state_dict"])
    
    wandb_logger = WandbLogger(
        project=args.project_name,
        name=f"dinohash_{args.lr}_{args.clean_weight}_{args.n_iter}",
        offline=args.offline,
        save_dir="./logs"
    )

    wandb_logger.experiment.config.update(vars(args))


    callbacks = [
        ModelCheckpoint(
            dirpath=f"./checkpoints",
            filename=f"dinov2_{args.model_name}{args.n_bits}.{args.lr}_{args.clean_weight}_{args.n_iter}_{{step}}",
            every_n_train_steps=args.val_freq,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='step'),
    ]
    
    trainer = L.Trainer(
        max_steps=args.steps,
        val_check_interval=args.val_freq,
        logger=wandb_logger,
        callbacks=callbacks,
        # gradient_clip_val=1.0,
        # gradient_clip_algorithm="norm",
        # precision="16-mixed",
        accelerator="auto",
        devices="auto",
        enable_progress_bar=True,
        log_every_n_steps=50
    )
    
    trainer.fit(model, datamodule=datamodule)
    
    print("Training completed!")
    
if __name__ == "__main__":
    main()