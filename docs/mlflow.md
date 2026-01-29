---
execute_via: python
---

# MLFlow

LaminDB can be integrated with [MLflow](https://mlflow.org/) to track model checkpoints as artifacts linked against training runs.

```python
# pip install lamindb torchvision lightning wandb
!lamin init --storage ./lamin-mlops
```

```python
import lightning as pl
import lamindb as ln
from lamindb.integrations import lightning as ll
import mlflow

from torch import utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from autoencoder import LitAutoEncoder

import logging

# Suppress unrelated logger messages
logging.getLogger("alembic").setLevel(logging.WARNING)
```

```{dropdown} Tracking models in both LaminDB and MLFlow
It is not always necessary to track all model parameters and metrics in both LaminDB and MLFlow.
However, if specific artifacts or runs should be queryable by specific model attributes such as, for example, the learning rate, then these attributes should be tracked.
Below, we show exemplary how to do that for the batch size and learning rate but the approach generalizes to more features.
```

```python
# define model run parameters, features, and labels so that validation passes later on
MODEL_CONFIG = {"batch_size": 32, "lr": 0.001}
hyperparameter = ln.Feature(name="Autoencoder hyperparameter", is_type=True).save()
hyperparams = ln.Feature.from_dict(MODEL_CONFIG, type=hyperparameter).save()

metrics_to_annotate = ["train_loss", "val_loss", "current_epoch"]
for metric in metrics_to_annotate:
    dtype = int if metric == "current_epoch" else float
    ln.Feature(name=metric, dtype=dtype).save()

# create all MLflow related features like 'mlflow_run_id'
ln.examples.mlflow.save_mlflow_features()

# create all lightning integration features like 'score'
ll.save_lightning_features()
```

```python
# track this notebook/script run so that all checkpoint artifacts are associated with the source code
ln.track(params=MODEL_CONFIG, project=ln.Project(name="MLflow tutorial").save())
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

## Monitor training with MLflow

Train our example model and track the training progress with `MLflow`.

```python
# enable MLFlow PyTorch autologging
mlflow.pytorch.autolog()
```

```python
with mlflow.start_run() as mlflow_run:
    train_dataset = MNIST(
        root="./data", train=True, download=True, transform=ToTensor()
    )
    val_dataset = MNIST(root="./data", train=False, download=True, transform=ToTensor())
    train_loader = utils.data.DataLoader(train_dataset, batch_size=32)
    val_loader = utils.data.DataLoader(val_dataset, batch_size=32)

    # create model
    autoencoder = LitAutoEncoder(hidden_size=32, bottleneck_size=16)

    # Create a LaminDB Lightning integration Checkpoint which also (optionally) annotates checkpoints by desired metrics
    lamindb_callback = ll.Checkpoint(
        dirpath=f"testmodels/mlflow/{mlflow_run.info.run_id}",
        features={
            "run": {
                "mlflow_run_id": mlflow_run.info.run_id,
                "mlflow_run_name": mlflow_run.info.run_name,
            },
            "artifact": {
                **{metric: None for metric in metrics_to_annotate}
            },  # auto-populated through callback
        },
    )

    # Train model
    trainer = pl.Trainer(
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
```

## MLflow and LaminDB user interfaces together

**MLflow and LaminDB runs:**

Both MLflow and LaminDB capture any runs together with run parameters.

| MLFlow experiment overview                                                                                                                                                                     | LaminHub run overview                                                                                                                                                            |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![MLFlow experiment overview UI](https://lamin-site-assets.s3.amazonaws.com/.lamindb/n0xxFoMRtZPiQ7VT0003.png)](https://lamin-site-assets.s3.amazonaws.com/.lamindb/n0xxFoMRtZPiQ7VT0003.png) | [![LaminHub run UI](https://lamin-site-assets.s3.amazonaws.com/.lamindb/aBXksZMr2VkX7Mfr0000.png)](https://lamin-site-assets.s3.amazonaws.com/.lamindb/aBXksZMr2VkX7Mfr0000.png) |

**MLflow run details and LaminDB artifact details:**

MLflow and LaminDB complement each other.
Whereas MLflow is excellent at capturing metrics over time, LaminDB excells at capturing lineage of input & output data and training checkpoints.

| MLFlow run view                                                                                                                                                              | LaminHub lineage view                                                                                                                                                                     |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![MLFlow runs](https://lamin-site-assets.s3.amazonaws.com/.lamindb/C0seowxsq4Du2B4T0002.png)](https://lamin-site-assets.s3.amazonaws.com/.lamindb/C0seowxsq4Du2B4T0002.png) | [![Laminhub lineage lineage](https://lamin-site-assets.s3.amazonaws.com/.lamindb/jLceaXyQf6WrFggW0000.png)](https://lamin-site-assets.s3.amazonaws.com/.lamindb/jLceaXyQf6WrFggW0000.png) |

Both frameworks display output artifacts that were generated during the run.
LaminDB further captures input artifacts, their origin and the associated source code.

| MLFlow artifact view                                                                                                                                                                | LaminHub artifact view                                                                                                                                                                |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![MLFlow artifact UI](https://lamin-site-assets.s3.amazonaws.com/.lamindb/k3ULj2AACQPASmUQ0000.png)](https://lamin-site-assets.s3.amazonaws.com/.lamindb/k3ULj2AACQPASmUQ0000.png) | [![LaminHub artifact UI](https://lamin-site-assets.s3.amazonaws.com/.lamindb/CQam6FY4V6DW65ek0000.png)](https://lamin-site-assets.s3.amazonaws.com/.lamindb/CQam6FY4V6DW65ek0000.png) |

All checkpoints are automatically annotated by the specified training metrics and MLflow run ID & name to keep both frameworks in sync:

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
