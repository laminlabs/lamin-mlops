import lamindb as ln
import lightning as pl
import mlflow
import argparse
import torch
from pathlib import Path

from torch import utils, optim, nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from lamindb.integrations import lightning as lnpl

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args()

MODEL_CONFIG = {"batch_size": args.batch_size, "lr": args.lr}

hyperparameter = ln.Feature(name="Autoencoder hyperparameter", is_type=True).save()
hyperparams = ln.Feature.from_dict(MODEL_CONFIG, str_as_cat=True)
for param in hyperparams:
    param.type = hyperparameter
    param.save()

ln.track(params=MODEL_CONFIG, project=ln.Project(name="MLflow runs").save())


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, hidden_size: int, bottleneck_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, bottleneck_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 28 * 28),
        )
        self.save_hyperparameters()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


artifact = ln.Artifact(
    "download_mnist",
    key="testdata/mnist",
    kind="dataset",
).save()

path = artifact.cache()

dataset = MNIST(path.as_posix(), transform=ToTensor())

mlflow.pytorch.autolog()

with mlflow.start_run() as mlflow_run:
    train_dataset = MNIST(
        root="./data", train=True, download=True, transform=ToTensor()
    )
    val_dataset = MNIST(root="./data", train=False, download=True, transform=ToTensor())

    train_loader = utils.data.DataLoader(train_dataset, batch_size=32)
    val_loader = utils.data.DataLoader(val_dataset, batch_size=32)

    autoencoder = LitAutoEncoder(32, 16)

    ckpt_dir = Path("model_checkpoints")
    ckpt_filename = "{mlflow_run.info.run_id}_last_epoch.ckpt"
    artifact_key = f"testmodels/mlflow/{mlflow_run.info.run_id}.ckpt"  # every run makes a new version of this artifact

    metrics = [
        ("epoch", int),
        ("global_step", int),
        ("train_loss", float),
        ("train_loss_step", float),
        ("val_loss", float),
        ("val_loss_step", float),
    ]

    # Create a LaminDB LightningCallback which also annotates check points by desired metrics
    metrics_to_annotate = ["train_loss", "val_loss"]
    for metric in metrics_to_annotate:
        ln.Feature(name=metric, dtype=float).save()
    lamindb_callback = lnpl.LightningCallback(
        path=ckpt_dir / ckpt_filename, key=artifact_key, annotate_by=metrics_to_annotate
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        limit_train_batches=3,
        max_epochs=5,
        callbacks=[lamindb_callback],
    )

    trainer.fit(
        model=autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # Register model_summary.txt
    local_model_summary_path = (
        f"{mlflow_run.info.artifact_uri.removeprefix('file://')}/model_summary.txt"
    )
    mlflow_model_summary_af = ln.Artifact(
        local_model_summary_path,
        key=local_model_summary_path,
        kind="model",
    ).save()

last_checkpoint_af = ln.Artifact.filter(
    key__startswith="testmodels/mlflow/", suffix__endswith="ckpt", is_latest=True
).last()
last_checkpoint_af.describe()

ln.finish()
