## Dummy Test Run

This document explains how to perform a quick dummy test run using `torchrun` and how to adjust `base.py` for a lightweight execution, as well as how to integrate parameter-freezing and hook-based gradient masking via `modify_trainables`.

### Prerequisites

* A Python virtual environment with all required dependencies installed (e.g., PyTorch, Transformers, etc.).
* Ensure you have a recent version of PyTorch (>= 1.9) that provides the `torchrun` entry point.
* Activate your environment before running the test:

  ```bash
  source /path/to/venv/bin/activate
  ```

### Running the Dummy Test

Use the following command to launch a single-process dummy test:

```bash
torchrun --nproc-per-node=1 dummy_test.py \
  --config=configs/train/contrastive_pretrain.yaml \
  --dtype=bf16
```

* `--nproc-per-node=1` launches one process (no multi-GPU).
* `--config=` points to your YAML configuration.
* `--dtype=bf16` specifies the mixed-precision datatype for the test.

### Adjustments in `base.py` for Dummy Run

For a lightweight dummy execution, you may need to comment out certain resource-intensive sections in `base.py`. For example:

```python
# self.dataloaders = self.get_dataloaders(config)
# self.scheduler = self.get_scheduler(config.train_args, self.optimizer, ds_config)
# self.tracker = self.get_trackers(config)
```

Commenting out these sections ensures the dummy script runs quickly, focusing only on a minimal forward/backward sanity check.

---

## Integrating Training-Parameter Modifications

To control which parameters are trainable and to apply gradient masking on your embeddings, include the `modify_trainables` helper in your code. You can import and call it before training begins (even in dummy mode) to ensure consistent behavior.

You can also control this by editing config 'contrastive_pretrain.yaml'

```python
import torch
from your_module.utils import modify_trainables

# After model initialization, apply one of the modes:
#   - 'unused_only': freeze all except rows 0–1000 of embeddings
#   - 'unused_and_rest': mask embeddings but leave other params trainable
#   - 'all': remove masks and make entire model trainable
modify_trainables(model, mode='freeze_unused_embeddings')

# Then proceed with dummy training loop or forward/backward check
outputs = model(dummy_input)
loss = compute_loss(outputs)
loss.backward()
```
