---
execute_via: python
---

# ClearML

```{info}
**How this differs from the W&B/MLflow tutorials**

The `wandb.md` and `mlflow.md` guides in this repository use a
feature-annotation strategy:

- create Lamin features (for example via `ln.examples.wandb.save_wandb_features()`
  or `ln.examples.mlflow.save_mlflow_features()`)
- pass run and artifact metadata into `ll.Checkpoint(features=...)`

The ClearML pattern below instead uses an artifact-observer strategy:

- subscribe to `ArtifactSavedEvent` and `ArtifactRemovedEvent`
- forward `checkpoint`, `config`, and `hparams` artifacts to ClearML using
  `event.storage_uri`

So W&B and MLflow here are metadata-enrichment tutorials, while ClearML is a
bridge-pattern sketch for deeper registry synchronization.
```

This page documents a draft integration approach and is intentionally not an
end-to-end runnable tutorial yet.

## Sketch

```{literalinclude} scripts/clearml_checkpoint_sketch.py
:language: python
```
