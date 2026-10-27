# Train a classification model

System-level single-label classification is supported by the PyTorch (`--pt`)
and PyTorch Exportable (`--pt-expt`) backends, including DPA4. Each frame belongs
to exactly one of `num_classes` mutually exclusive categories. This is not a
multi-label classifier or a discretization of continuous regression targets.

Set the fitting and loss types to `classification`:

```json
{
  "model": {
    "type": "DPA4",
    "type_map": ["H", "C", "N", "O"],
    "descriptor": {"rcut": 6.0, "channels": 32, "n_blocks": 2},
    "fitting_net": {
      "type": "classification",
      "num_classes": 3,
      "label_name": "phase",
      "neuron": [64, 64]
    }
  },
  "loss": {"type": "classification"}
}
```

This fragment supplements the usual learning-rate and training settings.
`label_name` defaults to `class`. Store one integer class ID per frame in
`set.*/phase.npy` (or `class.npy` for the default), with shape `(nframes, 1)`.
Labels must be finite integers from `0` through `num_classes - 1`; do not use
one-hot vectors. Standard coordinate, atom-type, and optional cell files are
still required. Use the same class-ID mapping in all training and test systems.

The fitting network outputs `num_classes` raw logits per atom. The model takes
their masked mean over real atoms to obtain system logits. These logits are
not normalized using class-label means or standard deviations. Training uses
mean cross-entropy; `accuracy` is the fraction of correctly classified frames.
Existing `property` fitting remains a regression task.

## Runnable toy example

`examples/property/classification` contains a small binary classification
example distinguishing two H2 bond-length ranges. These are synthetic labels
for checking the workflow, not physical molecular phases.

```sh
cd examples/property/classification
python make_data.py
dp --pt train input_dpa4.json
# Alternatively, in a separate run directory:
dp --pt-expt train input_dpa4.json
```

The resulting `model.ckpt.pt` can be loaded directly for inference:

```python
import numpy as np
from deepmd.infer import DeepEval

model = DeepEval("model.ckpt.pt")
coords = np.array([[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]]])
logits = model.eval(coords, None, [0, 0])[0]
classes = model.predict(coords, None, [0, 0])
probabilities = model.predict_proba(coords, None, [0, 0])
```

`eval` preserves raw logits, `predict` returns the largest-logit class ID, and
`predict_proba` applies softmax across classes. Atomic outputs, when requested
with `atomic=True`, are atomic logits, not independently supervised labels.
`dp test -m model.ckpt.pt -s data/valid` reports cross-entropy and accuracy,
weighted by the number of frames across systems.
