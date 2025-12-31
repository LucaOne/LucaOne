# LucaOne(LucaGPLM)
---

# LucaOne: a Foundation Model for Genomic and Protein Sequences

[![PyPI version](https://img.shields.io/badge/pip-v1.1.0-blue)](https://pypi.org/project/lucaone/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/your-username/LucaOne)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LucaOne** is a foundation model based on the **LucaGPLM** architecture, specifically engineered for biological sequences including DNA, Proteins, and RNA. This repository provides the refactored implementation that is fully compatible with the Hugging Face `transformers` ecosystem, supporting seamless integration for various downstream bioinformatics tasks.

## Key Features

- **Hugging Face Native**: Full support for `AutoModel`, `AutoModelForMaskedLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`, `AutoConfig`, and `AutoTokenizer`.
- **Unified Architecture**: Single model architecture handling multiple biological modalities.
- **Task-Specific Heads**:
    - `LucaGPLMModel`: For sequences embedding.
    - `LucaGPLMForMaskedLM`: For pre-training and sequence recovery.
    - `LucaGPLMForSequenceClassification`: For sequence-level tasks (e.g., protein family, solubility, or promoter prediction).
    - `LucaGPLMForTokenClassification`: For residue-level tasks (e.g., secondary structure, binding sites, or post-translational modifications).
- **Extensible**: Easily adaptable to custom downstream tasks using the standard `transformers` API.

## Installation   

```bash
pip install lucaone==1.1.0
pip install tokenizers==0.19.1
pip install transformers==4.41.2
```

You can install LucaOne directly from source:
```bash
git clone -b huggingface https://github.com/LucaOne/LucaOne.git
cd lucaone
pip install .
```

For development mode:
```bash
pip install -e .
```

## 🚀Quick Start

### 1. Feature Extraction/Embedding      
Extract high-dimensional embeddings for downstream analysis or training downstream tasks using LucaOne-Embedding:

Please refer to the code in `test/test_lucaone_embedding.py`.

### 2. MLM Pre-training and Sequence Recovery      
Continue to perform MLM pre-training or sequence recovery.

Please refer to the code in `test/test_lucaone_mlm.py`.

### 3. Sequence Classification
Predict properties for the entire sequence (e.g., Enzyme vs. Non-Enzyme):  

Supports `multi-class classification`, `binary classification`, `multi-label classification`, and `regression` tasks.

Please refer to the code in `test/test_lucaone_seq_classification.py`.  


### 4. Token Classification   
Predict properties for each residue/nucleotide (e.g., Secondary Structure, Binding Sites, and , Post-Translational Modifications):

Supports `multi-class classification`, `binary classification`, `multi-label classification`, and `regression` tasks.


Please refer to the code in `test/test_lucaone_token_classification.py`.

## Model Configuration

| Parameter | Description                                        | Default Value                |
| :--- |:---------------------------------------------------|:-----------------------------|
| `vocab_size` | Size of the dictionary                             | 39                           |
| `hidden_size` | Dimension of the hidden layers                     | 2560                         |
| `num_hidden_layers` | Number of Transformer layers                       | 20                           |
| `num_attention_heads` | Number of attention heads                          | 40                           |
| `position_embeddings` | -                                                  | ROPE                         |
| `alphabet` | Type of sequences (e.g., `gene`, `prot`, or `gene_prot`) | `DNA`, `RNA`, and `Protein`  |

## Weights Conversion

If you have legacy weights in `.pth` format, use the provided conversion script to migrate them to the Hugging Face format. This script maps the original state dictionary to the new `lucaone.` prefixed structure.

```bash
python scripts/convert_weights.py --input path/to/original.pth --output ./hf_model_dir
```

## Citation

If you use LucaOne in your research, please cite:

```bibtex
@article{lucaone2024,
  title={Generalized biological foundation model with unified nucleic acid and protein language.},
  author={He, Yong, Fang, P., Shan, Y. et al.},
  journal={Nat Mach Intell},
  year={2025},
  url={https://doi.org/10.1038/s42256-025-01044-4}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*For more information or issues, please open a GitHub issue or contact the maintainers at [sanyuan.hy@alibaba-inc.com/heyongcsat@gmail.com].*