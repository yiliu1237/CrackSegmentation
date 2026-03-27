from __future__ import division

import argparse
import os
from pathlib import Path

import cv2
import torch
import torch.optim as optim
import yaml
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import CrackDataset
from model import CrackModelDinoV3
from utils import ComputeIoU, CosineLRScheduler, dice_loss

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Train crack segmentation model.")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML.")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return resolve_experiment_config(config)


def resolve_experiment_config(config):
    """
    Resolve experiment naming and output dirs.
    Supports:
    - top-level `task_name`
    - `{task_name}` template in checkpoint.save_dir / logging.log_dir
    - default output dirs if omitted
    """
    task_name = config.get("task_name")
    ckpt_cfg = config.setdefault("checkpoint", {})
    log_cfg = config.setdefault("logging", {})

    if task_name:
        save_dir = ckpt_cfg.get("save_dir")
        log_dir = log_cfg.get("log_dir")

        if save_dir:
            ckpt_cfg["save_dir"] = save_dir.format(task_name=task_name)
        else:
            ckpt_cfg["save_dir"] = f"./checkpoints/{task_name}"

        if log_dir:
            log_cfg["log_dir"] = log_dir.format(task_name=task_name)
        else:
            log_cfg["log_dir"] = f"./logs/{task_name}"

    return config


def build_dataset_group(dataset_items, image_size, training, default_repeat_ratio):
    datasets = []
    for item in dataset_items:
        img_dir = item["images"]
        mask_dir = item["masks"]
        repeat_ratio = item.get("repeat_ratio", default_repeat_ratio)

        datasets.append(
            CrackDataset(
                image_size=(image_size, image_size),
                img_dir=img_dir,
                img_fnames=[path.name for path in Path(img_dir).glob("*") if path.is_file()],
                mask_dir=mask_dir,
                mask_fnames=[path.name for path in Path(mask_dir).glob("*") if path.is_file()],
                training=training,
                repeat_ratio=repeat_ratio,
            )
        )
    return ConcatDataset(datasets)


class Trainer(object):
    def __init__(self, rank, world_size, config):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.step = 0
        self.init_distributed()
        self.init_datasets()
        self.init_model()
        self.init_writer()
        self.train()
        self.cleanup()

    def init_distributed(self):
        self.log("Initializing distributed")
        os.environ["MASTER_ADDR"] = self.config["distributed"]["addr"]
        os.environ["MASTER_PORT"] = str(self.config["distributed"]["port"])
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)

    def init_datasets(self):
        self.log("Initializing dataset")
        image_size = self.config["training"]["image_size"]
        batch_size = self.config["training"]["batch_size"]
        num_workers = self.config["training"]["num_workers"]

        self.valid_dataset = build_dataset_group(
            dataset_items=self.config["datasets"]["val"],
            image_size=image_size,
            training=False,
            default_repeat_ratio=1,
        )

        self.train_dataset = build_dataset_group(
            dataset_items=self.config["datasets"]["train"],
            image_size=image_size,
            training=True,
            default_repeat_ratio=5,
        )

        self.datasampler_train = DistributedSampler(
            dataset=self.train_dataset,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=True,
        )

        self.dataloader_train = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=self.datasampler_train,
            prefetch_factor=2,
            pin_memory=True,
        )

        self.datasampler_valid = DistributedSampler(
            dataset=self.valid_dataset,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=True,
        )

        self.dataloader_valid = DataLoader(
            dataset=self.valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=self.datasampler_valid,
            prefetch_factor=2,
            pin_memory=True,
        )
        self.log("Initializing dataset finish")

    def init_model(self):
        self.log("Initializing model")
        model_cfg = self.config["model"]
        train_cfg = self.config["training"]
        ckpt_cfg = self.config["checkpoint"]

        self.model = CrackModelDinoV3(
            dinov3_arch=model_cfg["backbone"],
            dinov3_ckpt=model_cfg["pretrained_weights"],
        ).to(self.rank)

        if ckpt_cfg.get("resume_from"):
            self.log(f"Restoring from checkpoint: {ckpt_cfg['resume_from']}")
            self.log(
                self.model.load_state_dict(
                    torch.load(ckpt_cfg["resume_from"], map_location="cpu"),
                    strict=True,
                )
            )

        self.model_ddp = DDP(
            self.model,
            device_ids=[self.rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

        self.optimizer = optim.AdamW(
            [{"params": self.model.parameters(), "lr": train_cfg["learning_rate"]}]
        )

        self.lr_scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=train_cfg["epochs"],
            lr_min=train_cfg["learning_rate"] * train_cfg["min_lr_ratio"],
            t_mul=1.0,
            decay_rate=0.1,
            warmup_lr_init=train_cfg["learning_rate"] * train_cfg["warmup_lr_ratio"],
            warmup_t=train_cfg["warmup_epochs"],
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
        )

        self.scaler = GradScaler()

    def init_writer(self):
        if self.rank == 0:
            self.log("Initializing writer")
            self.writer = SummaryWriter(self.config["logging"]["log_dir"])

    def train(self):
        train_cfg = self.config["training"]
        log_cfg = self.config["logging"]
        ckpt_cfg = self.config["checkpoint"]

        imagenet_default_mean = (
            torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(self.rank).unsqueeze(0)
        )
        imagenet_default_std = (
            torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(self.rank).unsqueeze(0)
        )

        def denorm(x):
            return x * imagenet_default_std + imagenet_default_mean

        for epoch in range(1, train_cfg["epochs"] + 1):
            if self.rank == 0:
                self.log(f"Training epoch: {epoch}, LR: {self.lr_scheduler._get_lr(epoch)[0]:.6e}")
            self.datasampler_train.set_epoch(epoch)
            self.model.train()

            for image, mask in tqdm(self.dataloader_train, dynamic_ncols=True):
                image = image.to(self.rank, dtype=torch.float, non_blocking=True)
                mask = mask.to(self.rank, dtype=torch.float, non_blocking=True)
                pred_mask = self.model_ddp(image)
                loss = dice_loss(pred_mask, mask)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.rank == 0 and self.step % log_cfg["log_interval"] == 0:
                    self.writer.add_scalar("train_loss", loss, self.step)

                if self.rank == 0 and self.step % log_cfg["image_log_interval"] == 0:
                    self.writer.add_image("train_pred_mask", make_grid(pred_mask, nrow=16), self.step)
                    self.writer.add_image("train_mask", make_grid(mask, nrow=16), self.step)
                    self.writer.add_image("train_image", make_grid(denorm(image), nrow=16), self.step)
                self.step += 1

            self.lr_scheduler.step(epoch)

            if epoch % ckpt_cfg["save_interval"] == 0 or epoch >= train_cfg["epochs"] - 1:
                self.validate(epoch)
                self.save(epoch)

    def validate(self, epoch):
        if self.rank == 0:
            self.log(f"Validating at the start of epoch: {epoch}")
            self.model_ddp.eval()
            total_loss, total_count = 0, 0
            iou_mem = ComputeIoU()
            with torch.no_grad():
                for image, mask in tqdm(self.dataloader_valid, dynamic_ncols=True):
                    image = image.to(self.rank, dtype=torch.float, non_blocking=True)
                    mask = mask.to(self.rank, dtype=torch.float, non_blocking=True)
                    batch_size = image.size(0)
                    pred_mask = self.model(image)

                    iou_mem((torch.sigmoid(pred_mask.squeeze(1)) > 0.5).int(), mask.squeeze(1).int())
                    loss = dice_loss(pred_mask, mask)

                    total_loss += loss.item() * batch_size
                    total_count += batch_size
            miou = iou_mem.get_miou(ignore=0)
            self.log(f"Validation set mIoU: {miou}")
            avg_loss = total_loss / total_count
            self.log(f"Validation set average loss: {avg_loss}")

            self.writer.add_scalar("valid_loss", avg_loss, self.step)
            self.model_ddp.train()
        dist.barrier()

    def save(self, epoch):
        if self.rank == 0:
            save_dir = self.config["checkpoint"]["save_dir"]
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(save_dir, f"epoch-{epoch}.pth"))
            self.log("Model saved")
        dist.barrier()

    def cleanup(self):
        dist.destroy_process_group()

    def log(self, msg):
        print(f"[GPU{self.rank}] {msg}")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    world_size = torch.cuda.device_count()
    if world_size <= 0:
        raise RuntimeError("No CUDA devices found. This training script requires GPU for NCCL DDP.")

    mp.spawn(
        Trainer,
        nprocs=world_size,
        args=(world_size, config),
        join=True,
    )
