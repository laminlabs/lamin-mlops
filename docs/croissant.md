---
execute_via: python
---

# Croissant

[Croissant 🥐](https://github.com/mlcommons/croissant) is a high-level format building on [schema.org](https://schema.org) for machine learning datasets that combines metadata, resource file descriptions, data structure, and default ML semantics into a single file.
It works with existing datasets to make them easier to find, use, and support with tools.

Here, we demonstrate how LaminDB can be used to validate Croissant files:

```python
# pip install lamindb
!lamin init --storage ./test-lamin-croissant
```

```python
import lamindb as ln
import json

ln.track()
```

```python
croissant_path, dataset1_path = ln.examples.croissant.mini_immuno()
croissant_path
```

```python
with open(croissant_path) as f:
    dictionary = json.load(f)

print(json.dumps(dictionary, indent=2))
```

```python
dataset1_path
```

```python
artifact = ln.integrations.curate_from_croissant(croissant_path)
```

Project label, license, description, version tag, and file paths are automatically extracted from the Croissant file.
More metadata can be supported in the future.

```python
artifact.describe()
```

```python
ln.finish()
```
