---
execute_via: python
---

# Weights & Biases

LaminDB can be integrated with W&B to track the training process and associate datasets & parameters with models.

```python
# pip install lamindb torchvision lightning wandb
!lamin init --storage ./lamin-mlops
!wandb login
```

```python
import lightning as pl
import lamindb as ln
from lamindb.integrations import lightning as ll
import wandb

from torch import utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from autoencoder import LitAutoEncoder
```

```python
# define model run parameters, features, and labels so that validation passes later on
MODEL_CONFIG = {"hidden_size": 32, "bottleneck_size": 16, "batch_size": 32}
hyperparameter = ln.Feature(name="Autoencoder hyperparameter", is_type=True).save()
hyperparams = ln.Feature.from_dict(MODEL_CONFIG, type=hyperparameter).save()

metrics_to_annotate = ["train_loss", "val_loss", "current_epoch"]
for metric in metrics_to_annotate:
    dtype = int if metric == "current_epoch" else float
    ln.Feature(name=metric, dtype=dtype).save()

# create all Wandb related features like 'wandb_run_id'
ln.examples.wandb.save_wandb_features()

# create all lightning integration features like 'score'
ll.save_lightning_features()
```

```python
# track this notebook/script run so that all checkpoint artifacts are associated with the source code
ln.track(params=MODEL_CONFIG, project=ln.Project(name="Wandb tutorial").save())
```

## Define a model

We use a basic PyTorch Lightning autoencoder as an example model.

````{dropdown} Code of LitAutoEncoder
```{eval-rst}
.. literalinclude:: autoencoder.py
   :language: python
   :caption: Simple autoencoder model
```
````

## Query & download the MNIST dataset

We saved the MNIST dataset in a [curation notebook](/mnist) which now shows up in the Artifact registry:

```python
ln.Artifact.filter(kind="dataset").to_dataframe()
```

Let's get the dataset:

```python
mnist_af = ln.Artifact.get(key="testdata/mnist")
mnist_af
```

And download it to a local cache:

```python
path = mnist_af.cache()
path
```

Create a PyTorch-compatible dataset:

```python
mnist_dataset = MNIST(path.as_posix(), transform=ToTensor())
mnist_dataset
```

## Monitor training with wandb

Train our example model and track the training progress with `wandb`.

```python
from lightning.pytorch.loggers import WandbLogger

# create the data loader
train_dataset = MNIST(root="./data", train=True, download=True, transform=ToTensor())
val_dataset = MNIST(root="./data", train=False, download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(train_dataset, batch_size=32)
val_loader = utils.data.DataLoader(val_dataset, batch_size=32)

# init model
autoencoder = LitAutoEncoder(
    MODEL_CONFIG["hidden_size"], MODEL_CONFIG["bottleneck_size"]
)

# initialize the logger
wandb_logger = WandbLogger(project="lamin")

# add batch size to the wandb config
wandb_logger.experiment.config["batch_size"] = MODEL_CONFIG["batch_size"]
```

```python
# Create a LaminDB LightningCallback which also (optionally) annotates checkpoints by desired metrics
wandb_logger.experiment.id
lamindb_callback = ll.Checkpoint(
    dirpath=f"testmodels/wandb/{wandb_logger.experiment.id}",
    features={
        "run": {
            "wandb_run_id": wandb_logger.experiment.id,
            "wandb_run_name": wandb_logger.experiment.name,
        },
        "artifact": {
            **{metric: None for metric in metrics_to_annotate}
        },  # auto-populated through callback
    },
)

# train model
trainer = pl.Trainer(
    limit_train_batches=3,
    max_epochs=5,
    logger=wandb_logger,
    callbacks=[lamindb_callback],
)
trainer.fit(
    model=autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader
)
```

```python
wandb_logger.experiment.name
```

```python
wandb.finish()
```

## W&B and LaminDB user interfaces together

**W&B and LaminDB runs:**

Both W&B and LaminDB capture any runs together with run parameters.

| W&B experiment overview                                                                                                                                                                     | LaminHub run overview                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![W&B experiment overview UI](https://lamin-site-assets.s3.amazonaws.com/.lamindb/DkMfwknGEBZ0EdTf0000.png)](https://lamin-site-assets.s3.amazonaws.com/.lamindb/DkMfwknGEBZ0EdTf0000.png) | [![LaminHub run UI](https://lamin-site-assets.s3.amazonaws.com/.lamindb/wpfQM12SXxY7owqR0000.png)](https://lamin-site-assets.s3.amazonaws.com/.lamindb/wpfQM12SXxY7owqR0000.png) |

**W&B run details and LaminDB artifact details:**

W&B and LaminDB complement each other.
Whereas W&B is excellent at capturing metrics over time, LaminDB excells at capturing lineage of input & output data and training checkpoints.

| W&B run view                                                                                                                                                              | LaminHub run view                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![W&B runs](https://lamin-site-assets.s3.amazonaws.com/.lamindb/Qunk9d9YvAcEhGjI0000.png)](https://lamin-site-assets.s3.amazonaws.com/.lamindb/Qunk9d9YvAcEhGjI0000.png) | [![Laminhub run lineage](https://lamin-site-assets.s3.amazonaws.com/.lamindb/oc4qSs8xvDjSw5g90000.png)](https://lamin-site-assets.s3.amazonaws.com/.lamindb/oc4qSs8xvDjSw5g90000.png) |

Both frameworks display output artifacts that were generated during the run.
LaminDB further captures input artifacts, their origin and the associated source code.

| W&B artifact view                                                                                                                                                                | LaminHub artifact view                                                                                                                                                                |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![W&B artifact UI](https://lamin-site-assets.s3.amazonaws.com/.lamindb/I20YlrvMhvvqIwqG0000.png)](https://lamin-site-assets.s3.amazonaws.com/.lamindb/I20YlrvMhvvqIwqG0000.png) | [![LaminHub artifact UI](https://lamin-site-assets.s3.amazonaws.com/.lamindb/rFHt5FeXgWp9nrtz0000.png)](https://lamin-site-assets.s3.amazonaws.com/.lamindb/rFHt5FeXgWp9nrtz0000.png) |

All checkpoints are automatically annotated by the specified training metrics and W&B run ID & name to keep both frameworks in sync:

```python
last_checkpoint_af = (
    ln.Artifact.filter(is_best_model=True)
    .filter(suffix__endswith="ckpt", is_latest=True)
    .last()
)
last_checkpoint_af.describe()
```

To reuse the checkpoint later:

```python
last_checkpoint_af.cache()
```

```python
last_checkpoint_af.view_lineage()
```

Features associated with a whole training run are annotated on a run level:

```python
ln.context.run.features
```

```python
ln.finish()
```
