---
execute_via: python
---

# Curate MNIST

```python
# pip install lamindb torch torchvision lightning
!lamin init --storage ./lamin-mlops
```

```python
import lamindb as ln
from pathlib import Path

ln.track()
```

Download the MNIST dataset and save it in LaminDB to keep track of the training data that is associated with our model.

```python
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

dataset = MNIST(Path.cwd() / "download_mnist", download=True, transform=ToTensor())
```

```python
# no need for the zipped files
!rm -r download_mnist/MNIST/raw/*.gz
!ls -r download_mnist/MNIST/raw
```

```python
training_data_artifact = ln.Artifact(
    "download_mnist/",
    key="testdata/mnist",
    kind="dataset",
    description="Complete MNIST dataset directory containing training and test data",
).save()
training_data_artifact
```

```python
ln.finish()
```
