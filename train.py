import torch
from torch.utils.data import Dataset, DataLoader
import os
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import sys
from tqdm import tqdm
from PIL import Image
import csv
from model import GenerativeImage2TextModel, get_image_encoder, get_git_model
from config import get_config
from distributed import init_distributed_device
from dataclass import DistArgs
from torch.nn.parallel import DistributedDataParallel
import wandb

Image.MAX_IMAGE_PIXELS = 1000000000


class ConceptualDataset(Dataset):
    def __init__(self, data_dir: str, suffix="train", preprocess=None):
        super().__init__()
        self.data_dir = data_dir
        self.suffix = suffix
        self.preprocess = preprocess
        self.image_dir = os.path.join(data_dir, suffix)
        self.captions = self.load_captions()
        self.image_ids = list(self.captions.keys())
        self.image_ids.sort()
        self.image_id2idx = {image_id: idx for idx, image_id in enumerate(self.image_ids)}
        self.idx2image_id = {idx: image_id for idx, image_id in enumerate(self.image_ids)}

    def load_captions(self):
        captions = {}
        if self.suffix == "train":
            tsv_path = f"{self.data_dir}/Train_GCC-training.tsv"
        else:
            tsv_path = f"{self.data_dir}/Validation_GCC-1.1.0-Validation.tsv"

        with open(tsv_path, "r") as f:
            read_tsv = csv.reader(f, delimiter="\t")
            for i, row in enumerate(read_tsv):
                caption, _ = row
                image_id = f"{i:08d}"
                captions[image_id] = caption
        return captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_id = self.idx2image_id[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = self.load_image(image_path)
        caption = self.captions[image_id]
        return image, caption

    def load_image(self, image_path: str):
        if not os.path.exists(image_path):
            image = torch.zeros(3, 224, 224)

        try:
            image = Image.open(image_path)
            image = self.preprocess(image)
        except:  # PIL.UnidentifiedImageError: cannot identify image file '../data/conceptual/train/01424292.jpg'
            image = torch.zeros(3, 224, 224)

        return image


def train(dataset: ConceptualDataset, model: GenerativeImage2TextModel, args):
    dist_args = DistArgs()
    device = init_distributed_device(dist_args)
    print(f"device: {device}")
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[device], output_device=device)
    if args.report_to_wandb and dist_args.rank == 0:
        wandb.init(project="generative image-to-text")
        wandb.config.update(args)
        wandb.watch(model)

    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=dist_args.world_size,
        rank=dist_args.rank,
        shuffle=True,
    )
    train_dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=4, sampler=train_sampler)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    for epoch in range(epochs):
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader))
        for idx, (images, captions) in enumerate(train_dataloader):
            model.zero_grad()
            images = images.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images, captions)
            loss = outputs["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            optimizer.zero_grad()

            progress.set_postfix({"loss": loss.item()})
            progress.update()

            if (idx + 1) % 100 == 0:
                if args.report_to_wandb and dist_args.rank == 0:
                    wandb.log({"loss": loss.item()})
                    wandb.log({"lr": scheduler.get_last_lr()[0]})

            if (idx + 1) % 10000 == 0:
                if dist_args.rank == 0:
                    print(f">>> Saving model at epoch {epoch} step {idx}")
                    torch.save(
                        model.module.state_dict(),
                        os.path.join(args.output_dir, f"model_{epoch}_{idx}.pt"),
                    )
        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            if dist_args.rank == 0:
                print(f">>> Saving model at epoch {epoch}")
                torch.save(
                    model.module.state_dict(),
                    os.path.join(args.output_dir, f"model_{epoch}.pt"),
                )
    return model


def main():
    args, _ = get_config()
    _, preprocess = get_image_encoder(args)
    dataset = ConceptualDataset(args.data_dir, suffix="train", preprocess=preprocess)
    model = get_git_model(args)
    train(dataset, model, args)


if __name__ == "__main__":
    main()
