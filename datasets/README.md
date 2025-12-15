# Downloaded Datasets

This directory stores datasets for studying jailbreaks and nonsense-style prompts. Data files are kept out of git via `.gitignore`; small samples are included for documentation.

## Dataset 1: Harmful Behaviors
- **Source**: `mlabonne/harmful_behaviors` on Hugging Face
- **Size**: 416 train, 104 test examples
- **Format**: Parquet (saved with 🤗 `datasets`)
- **Task**: Jailbreak evaluation (harmful and harmless instructions)
- **License**: MIT
- **Location**: `datasets/harmful_behaviors/`

### Download Instructions
Using Hugging Face:
```python
from datasets import load_dataset
ds = load_dataset("mlabonne/harmful_behaviors")
ds.save_to_disk("datasets/harmful_behaviors")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/harmful_behaviors")
```

### Sample
See `datasets/harmful_behaviors/samples/train_first5.json` for first 5 training rows.

### Notes
- Includes paired harmless/harmful behaviors useful for alignment or refusal evaluation.

## Dataset 2: Augmented AdvBench v3 (filtered)
- **Source**: `Baidicoot/augmented_advbench_v3_filtered` on Hugging Face
- **Size**: 5,230 train examples
- **Format**: Parquet (saved with 🤗 `datasets`)
- **Task**: Jailbreak-style adversarial prompts with augmentations
- **License**: Not specified on card (check upstream HF page)
- **Location**: `datasets/augmented_advbench_v3_filtered/`

### Download Instructions
Using Hugging Face:
```python
from datasets import load_dataset
ds = load_dataset("Baidicoot/augmented_advbench_v3_filtered")
ds.save_to_disk("datasets/augmented_advbench_v3_filtered")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/augmented_advbench_v3_filtered")
```

### Sample
See `datasets/augmented_advbench_v3_filtered/samples/train_first5.json`.

### Notes
- Augmented AdvBench prompts provide diverse jailbreak instructions for stress-testing safety filters.
