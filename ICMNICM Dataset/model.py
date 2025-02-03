import torch
from torch import nn
import lightning.pytorch as pl
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
from CMRFormer.video_transformer import SpaceTimeTransformer
import config


class icm_pred(pl.LightningModule):
    """
    Extracts the video embeddings using a pre-trained video transformer.
    ICM/NICM classification.
    """

    def __init__(self, n_batches=None, n_epochs=None, **kwargs):
        super().__init__()

        # Parameters
        self.n_batches = n_batches
        self.n_epochs = n_epochs
        self.batch_size = kwargs["batch_size"]
        self.lr = kwargs["lr"]
        self.emb_dim = kwargs["emb_dim"]
        self.hidden_dim1 = kwargs["hidden_dim1"]
        self.hidden_dim2 = kwargs["hidden_dim2"]
        self.num_classes = kwargs["num_classes"]
        self.dropout_ratio = kwargs["dropout_ratio"]

        # Architecture
        # SpaceTimeTransformer
        self.encoder = SpaceTimeTransformer(
            num_frames=32,
            time_init="zeros",
            attention_style="frozen-in-time",
        )
        self.encoder.head = nn.Identity()
        self.weights = torch.load(
            "/data/aiiih/projects/nakashm2/multimodal/Video_text_retrieval/exps/models/MSRVTTjsfusion_4f_stformer_pt-im21k/1019_093959/vid_encoder.pth"
        )
        self.encoder.load_state_dict(self.weights)

        # Freeze encoder
        for param in list(self.encoder.parameters()):
            param.requires_grad = False

        # Discrete disease classifier
        self.fc1 = nn.Linear(self.emb_dim, self.hidden_dim1)
        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc3 = nn.Linear(self.hidden_dim2, self.num_classes)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim1)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_ratio)

        # Loss
        self.loss = nn.CrossEntropyLoss()

        # Logs
        self.processed_batches = 0
        self.train_loss, self.val_loss = 0, 0
        self.train_step_outputs, self.val_step_outputs = ([], [])
        print("Model Initialized!")

    def configure_optimizers(self):
        # Optimizer
        self.trainable_params = [
            param for param in self.parameters() if param.requires_grad
        ]
        optimizer = AdamW(self.trainable_params, lr=self.lr)

        # Scheduler
        warmup_steps = self.n_batches // 3
        total_steps = self.n_batches * self.n_epochs - warmup_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
        return [optimizer], [scheduler]

    def forward(self, videos):
        features = self.encoder(videos)
        x = self.fc1(features)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits

    def training_step(self, batch, batch_idx):
        videos = batch["videos"]
        labels = batch["labels"]

        logits = self(videos)
        loss = self.loss(logits, labels)
        if torch.isnan(loss):
            print(batch)
        logs = {"train_batch_loss": loss}
        self.train_step_outputs.append(logs)
        self.log("train_batch_loss", loss, prog_bar=True, logger=True, sync_dist=True)

        # Average losses calculation
        self.processed_batches += 1
        self.train_loss += loss

        if self.processed_batches % config.log_every_n_steps == 0:
            train_step_avg_loss = self.train_loss / self.processed_batches
            self.log(
                "train_step_avg_loss",
                train_step_avg_loss,
                logger=True,
                sync_dist=True,
            )
            self.processed_batches = 0
            self.train_loss = 0
        return loss

    def on_train_epoch_end(self):
        train_epoch_avg_loss = (
            torch.stack([x["train_batch_loss"] for x in self.train_step_outputs])
            .mean()
            .detach()
            .cpu()
            .numpy()
        )
        self.log(
            "train_epoch_avg_loss",
            train_epoch_avg_loss.item(),
            logger=True,
            sync_dist=True,
        )
        print(
            "train_epoch:",
            self.current_epoch,
            "train_epoch_avg_loss:",
            train_epoch_avg_loss,
        )
        self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        videos = batch["videos"]
        labels = batch["labels"]

        logits = self(videos)
        loss = self.loss(logits, labels)
        logs = {"val_batch_loss": loss}
        self.val_step_outputs.append(logs)
        self.log("val_batch_loss", loss, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        val_epoch_avg_loss = (
            torch.stack([x["val_batch_loss"] for x in self.val_step_outputs])
            .mean()
            .detach()
            .cpu()
            .numpy()
        )
        self.log(
            "val_epoch_avg_loss",
            val_epoch_avg_loss.item(),
            logger=True,
            sync_dist=True,
        )

        print(
            "val_epoch:",
            self.current_epoch,
            "val_epoch_avg_loss:",
            val_epoch_avg_loss,
        )
        self.val_step_outputs.clear()
