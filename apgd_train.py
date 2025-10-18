import argparse
import os
from typing import Optional, List, Dict, Any

from PIL import Image
import torch
from torch.optim import AdamW
from adamwsp import AdamWSP
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import webdataset as wds

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from hashes.dinohash import DINOHash, preprocess
from apgd_attack import APGDAttack, criterion_loss

torch.manual_seed(0)
np.random.seed(0)
torch.set_float32_matmul_precision("medium")

class DataModule(L.LightningDataModule):    
    def __init__(
        self, 
        batch_size: int = 200, 
        split_shards: float = 1,
        num_workers: int = 11
    ):
        super().__init__()
        self.batch_size = batch_size
        self.split_shards = split_shards
        self.num_workers = num_workers
        self.save_hyperparameters()
        
    def setup(self, stage: Optional[str] = None) -> None:
        num_shards = 1159  # from {00000000..00001158}
        split_point = num_shards - self.split_shards

        train_pattern = f"/mnt/unified_dataset/shards/{{00000000..{split_point-1:08d}}}.tar"
        val_pattern = f"/mnt/unified_dataset/shards/{{{split_point:08d}..{num_shards-1:08d}}}.tar"

        self.train_dataset = (
            wds.WebDataset(train_pattern, handler=wds.handlers.warn_and_continue)
            .shuffle(10000)
            .decode("pil")
            .to_tuple("jpg")
            .map(lambda x: preprocess(x[0]))
        )

        self.val_dataset = (
            wds.WebDataset(val_pattern, handler=wds.handlers.warn_and_continue, empty_check=False)
            .decode("pil")
            .to_tuple("jpg")
            .map(lambda x: preprocess(x[0]))
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
        )
        
    def complete_dataloader(self) -> DataLoader:
        return DataLoader(
            self.complete_dataset, 
            batch_size=self.batch_size, 
        )


class AdversarialDINOHashModule(L.LightningModule):    
    def __init__(
        self,
        model_name: str = "vits14_reg",
        version: str = 'v2',
        n_bits: int = 96,
        epsilon: float = 8/255,
        n_iter: int = 20,
        attack_schedule: bool = False,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        warmup: int = 1400,
        steps: int = 20000,
        clean_weight: float = 500,
        uap: bool = False,
        use_amp: bool = True,
        gradient_clip_val: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.attack_schedule = attack_schedule
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.steps = steps
        self.clean_weight = clean_weight
        self.use_amp = use_amp
        self.gradient_clip_val = gradient_clip_val
        
        self.clean_dinohash = DINOHash(
            model=model_name, 
            pca_dims=n_bits, 
            prod_mode=False,
            version=version
        )
        self.adversarial_dinohash = DINOHash(
            model=model_name, 
            pca_dims=n_bits, 
            prod_mode=False,
            version=version
        )
        
        for param in self.clean_dinohash.dino.parameters():
            param.requires_grad = False
        self.clean_dinohash.dino.eval()
        
        for param in self.adversarial_dinohash.dino.parameters():
            param.requires_grad = True
            
        self.apgd = APGDAttack(
            dinohash=self.adversarial_dinohash, 
            eps=self.epsilon,
            uap=uap
        )
        
        self.automatic_optimization = False
        
        # Initialize GradScaler for mixed precision
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = AdamWSP(
            self.adversarial_dinohash.dino.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95)
        )
        
        def lr_lambda(step: int) -> float:
            if step < self.warmup:
                return step / self.warmup
            else:
                progress = (step - self.warmup) / (self.steps - self.warmup)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def get_scheduled_n_iter(self, current_step: int) -> int:
        if not self.attack_schedule:
            return self.n_iter
        progress = current_step / self.steps
        scheduled_n_iter = int(1 + self.n_iter * progress)
        
        return scheduled_n_iter
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        
        optimizer.zero_grad(set_to_none=True)
        
        batch = batch.to(self.device)
        
        # Get clean logits (no autocast for frozen model)
        with torch.no_grad():
            logits = self.clean_dinohash.hash(batch, differentiable=False, logits=True).float()
        
        # Get the scheduled number of iterations based on current step
        scheduled_n_iter = self.get_scheduled_n_iter(self.global_step)
        
        eps = np.random.uniform(0.5, 1.5) * self.epsilon
        
        # Clean loss computation with autocast
        if self.use_amp:
            with torch.cuda.amp.autocast():
                _, clean_loss, adversarial_logits = criterion_loss(
                    batch, logits, self.adversarial_dinohash.hash, loss="target bce"
                )
                clean_loss = self.clean_weight * clean_loss.mean()
            
            # Scale and backward
            self.scaler.scale(clean_loss).backward()
        else:
            _, clean_loss, adversarial_logits = criterion_loss(
                batch, logits, self.adversarial_dinohash.hash, loss="target bce"
            )
            clean_loss = self.clean_weight * clean_loss.mean()
            clean_loss.backward()

        # Generate adversarial examples (outside autocast for stability)
        with torch.no_grad():
            adv_images, _ = self.apgd.attack_single_run(
                batch, adversarial_logits.detach(), scheduled_n_iter, log=False, eps=eps
            )

        if self.use_amp:
            with torch.cuda.amp.autocast():
                adv_hashes, adv_loss, _ = criterion_loss(
                    adv_images, adversarial_logits.detach(), self.adversarial_dinohash.hash, loss="target bce"
                )
                adv_loss = adv_loss.mean()
            
            self.scaler.scale(adv_loss).backward()
            self.scaler.unscale_(optimizer)
        else:
            adv_hashes, adv_loss, _ = criterion_loss(
                adv_images, adversarial_logits.detach(), self.adversarial_dinohash.hash, loss="target bce"
            )
            adv_loss = adv_loss.mean()
            adv_loss.backward()
        
        if self.gradient_clip_val > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.adversarial_dinohash.dino.parameters(), 
                self.gradient_clip_val
            )
            self.log('train/grad_norm', grad_norm, on_step=True)
        
        # Optimizer step with scaler
        if self.use_amp:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        
        scheduler.step()

        total_loss = adv_loss + clean_loss
        
        # Metrics computation (no gradient needed)
        with torch.no_grad():
            hashes = (logits >= 0).float()
            accuracy = (adv_hashes - hashes).abs().mean()
        
        self.log('train/total_loss', total_loss, on_step=True)
        self.log('train/adv_loss', adv_loss, on_step=True)
        self.log('train/clean_loss', clean_loss / max(self.clean_weight, 1), on_step=True)
        self.log('train/accuracy', accuracy * 100, on_step=True, prog_bar=True)
        self.log('train/lr', optimizer.param_groups[0]['lr'], on_step=True)
        
        if self.use_amp:
            self.log('train/scale', self.scaler.get_scale(), on_step=True)
        
        return total_loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        self.adversarial_dinohash.dino.eval()
        
        with torch.no_grad():
            adversarial_logits = self.adversarial_dinohash.hash(batch, differentiable=False, logits=True).float()
            
            adv_images, _ = self.apgd.attack_single_run(
                batch, adversarial_logits, n_iter=50, eps=self.epsilon
            )
            
            logits = self.adversarial_dinohash.hash(batch, differentiable=False, logits=True).float()
            hashes = (logits >= 0).float()

            clean_hashes = self.clean_dinohash.hash(batch).float()
            clean_accuracy = (clean_hashes - hashes).abs().mean()

            adv_hashes = self.adversarial_dinohash.hash(adv_images).float()
            attack_accuracy = (adv_hashes - clean_hashes).abs().mean()
        
        self.log('val/attack_strength', attack_accuracy * 100, on_epoch=True, prog_bar=True)
        self.log('val/clean_error', clean_accuracy * 100, on_epoch=True, prog_bar=True)

def main():
    parser = argparse.ArgumentParser(description='Adversarial neural collision attack with Lightning')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for processing images')
    parser.add_argument('--n_iter', type=int, default=20, help='Final number of attack iterations')
    parser.add_argument('--version', type=str, default='v2', choices=['v2', 'v3'], help='DINO version')
    parser.add_argument('--attack_schedule', action='store_true', help='Use linear attack schedule')
    parser.add_argument('--epsilon', type=float, default=8/255, help='Maximum perturbation (L∞ norm bound)')
    parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--warmup', type=int, default=1400, help='Number of warmup steps')
    parser.add_argument('--steps', type=int, default=20000, help='Number of steps')
    parser.add_argument('--clean_weight', type=float, default=500, help='Weight of clean loss')
    parser.add_argument('--val_freq', type=int, default=500, help='Validation frequency')
    parser.add_argument('--resume_path', type=str, default=None, help='Resume path')
    parser.add_argument('--n_bits', type=int, default=96, help='Number of PCA components for DINOHash')
    parser.add_argument('--model_name', type=str, default="vits14_reg", help='Model backbone for DINO')
    parser.add_argument('--uap', action='store_true', help='Use universal perturbation as start')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--no_amp', action='store_true', help='Disable automatic mixed precision')
    
    parser.add_argument('--project_name', type=str, default='adversarial-dinohash', help='Wandb project name')
    parser.add_argument('--experiment_name', type=str, default=None, help='Wandb experiment name')
    parser.add_argument('--offline', action='store_true', help='Run wandb in offline mode')
    
    args = parser.parse_args()
    
    datamodule = DataModule(
        batch_size=args.batch_size,
        split_shards=1,
        num_workers=11
    )
    
    model = AdversarialDINOHashModule(
        model_name=args.model_name,
        version=args.version,
        n_bits=args.n_bits,
        epsilon=args.epsilon,
        n_iter=args.n_iter,
        attack_schedule=args.attack_schedule,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup=args.warmup,
        steps=args.steps,
        clean_weight=args.clean_weight,
        uap=args.uap,
        use_amp=not args.no_amp,
        gradient_clip_val=args.gradient_clip_val
    )
    
    if args.resume_path:
        checkpoint = torch.load(args.resume_path)
        model.adversarial_dinohash.dino.load_state_dict(checkpoint)
    
    wandb_logger = WandbLogger(
        project=args.project_name,
        name=args.experiment_name or f"dinohash_{args.lr}_{args.clean_weight}_{args.n_iter}",
        offline=args.offline,
        save_dir="./logs"
    )
    
    callbacks = [
        ModelCheckpoint(
            dirpath=f"./checkpoints",
            filename=f"dino{args.version}_{args.model_name}{args.n_bits}.{args.lr}_{args.clean_weight}_{args.n_iter}_{{step}}",
            monitor="val/attack_strength",
            mode="max",
            save_top_k=1,
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
        accelerator="auto",
        devices="auto",
        enable_progress_bar=True,
        log_every_n_steps=50
    )
    
    trainer.fit(model, datamodule=datamodule)
    
    print("Training completed!")


if __name__ == "__main__":
    main()