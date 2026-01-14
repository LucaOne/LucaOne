---
license: mit
tags:
  - LucaOne
  - Biological Foundation Model
  - Unified Nucleic Acid and Protein Language Model
  - Biology
  - AI4Science
  - AI4Biology
  - Bio
  - 1.1.1
language:
  - en
---

# LucaOne/LucaGPLM

LucaOne/LucaGPLM - The LUCA Gene-Protein language model.

## Installation

You can install the package from source using pip:

```bash
pip install lucaone==1.1.1
pip install tokenizers==0.19.1
pip install transformers==4.41.2
```

## Usage

Please refer to the `huggingface` branch of LucaOne: https://github.com/LucaOne/LucaOne.

### 1. Feature Extraction/Embedding    
Extract high-dimensional embeddings for downstream analysis or training downstream tasks using LucaOne-Embedding.     

```python
import torch
import lucaone
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer

# model_id
model_id = "LucaGroup/LucaOne-default-step36M"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    force_download=True
)

model = AutoModel.from_pretrained(
    model_id,
    task_level="token_level",
    task_type="embedding",
    trust_remote_code=True,
    force_download=True
)
print(model)
print("*" * 50)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# nucleotide sequence
nucleotide_sequence = "ATGCGTACGTTAGC"
print("Nucleotide sequence len: %d" % len(nucleotide_sequence))

# nucleotide sequence embedding
print("Processing Nucleotide Sequence...")
nucleotide_inputs = tokenizer(
    nucleotide_sequence,
    # note: gene sequence(for DNA or RNA)
    seq_type="gene",
    return_tensors="pt",
    add_special_tokens=True
)
new_nucleotide_inputs = {}
for item in nucleotide_inputs.items():
    new_nucleotide_inputs[item[0]] = item[1].to(device)
nucleotide_inputs = new_nucleotide_inputs
print("Nucleotide inputs:")
print(nucleotide_inputs)

with torch.no_grad():
    nucleotide_outputs = model(**nucleotide_inputs)
    # last hidden matrix as embedding matrix: [batch_size, seq_len + 2, hidden_size]
    nucleotide_last_hidden = nucleotide_outputs.last_hidden_state
    # mean pooling
    mean_nucleotide_embedding = nucleotide_last_hidden[0, 1:-1, :].mean(dim=1)
    # cls pooling
    cls_nucleotide_embedding = nucleotide_last_hidden[0, 0, :]
print(f"Nucleotide Embedding Shape: {nucleotide_last_hidden.shape}")
print("Nucleotide Embedding(Matrix, Include [CLS] and [SEP]):")
print(nucleotide_last_hidden)
print("Nucleotide Embedding(Mean Pooling Vector):")
print(mean_nucleotide_embedding)
print("Nucleotide Embedding(CLS Pooling Vector):")
print(cls_nucleotide_embedding)
print("*" * 50)

# Protein Sequence
protein_sequence = "MKTLLILTAVVLL"
print("Protein sequence len: %d" % len(nucleotide_sequence))

print("Processing Protein Sequence...")
prot_inputs = tokenizer(
    protein_sequence,
    # note: protein sequence
    seq_type="prot",
    return_tensors="pt",
    add_special_tokens=True
)
new_prot_inputs = {}
for item in prot_inputs.items():
    new_prot_inputs[item[0]] = item[1].to(device)
prot_inputs = new_prot_inputs
print("Protein inputs:")
print(prot_inputs)

with torch.no_grad():
    prot_outputs = model(**prot_inputs)
    # last hidden matrix as embedding matrix: [batch_size, seq_len + 2, hidden_size]
    prot_last_hidden = prot_outputs.last_hidden_state
    # mean pooling
    mean_prot_embedding = prot_last_hidden[:, 1:-1, :].mean(dim=1)
    # cls pooling
    cls_prot_embedding = prot_last_hidden[:, 0, :]
print(f"Protein Embedding Shape: {prot_last_hidden.shape}")
print("Protein Embedding(Matrix, Include [CLS] and [SEP]):")
print(prot_last_hidden)
print("Protein Embedding(Mean Pooling Vector):")
print(mean_prot_embedding)
print("Protein Embedding(CLS Pooling Vector):")
print(cls_prot_embedding)
print("*" * 50)
```

### 2. MLM Pre-training and Sequence Recovery
Continue to perform MLM pre-training or sequence recovery.    

```python
import torch
import lucaone
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer

# model_id
model_id = "LucaGroup/LucaOne-default-step36M"

model = AutoModelForMaskedLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    force_download=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    force_download=True
)
print(model)
print("*" * 50)

# finetune all parameters
for param in model.parameters():
    param.requires_grad = True

# create dataset and trainer for training...
```
### 3. Sequence Classification
Predict properties for the entire sequence (e.g., Enzyme vs. Non-Enzyme).   

Supports `multi-class classification`, `binary classification`, `multi-label classification`, and `regression` tasks.    

```python
import torch
import lucaone
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# model_id
model_id = "LucaGroup/LucaOne-default-step36M"

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    task_level="seq_level",
    task_type="multi_class",
    classifier_num_labels=4,
    trust_remote_code=True,
    force_download=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    force_download=True
)
print(model)
print("*" * 50)

# finetune all parameters
for param in model.parameters():
    param.requires_grad = True

# create dataset and trainer for training...
```
### 4. Token Classification
Predict properties for each residue/nucleotide (e.g., Secondary Structure, Binding Sites, and , Post-Translational Modifications).

Supports `multi-class classification`, `binary classification`, `multi-label classification`, and `regression` tasks.

```python
import torch
import lucaone
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer

# model_id
model_id = "LucaGroup/LucaOne-default-step36M"

model = AutoModelForTokenClassification.from_pretrained(
    model_id,
    task_level="token_level",
    task_type="binary_class",
    classifier_num_labels=2,
    trust_remote_code=True,
    force_download=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    force_download=True
)
print(model)
print("*" * 50)

# finetune all parameters
for param in model.parameters():
    param.requires_grad = True

# create dataset and trainer for training...
```
## Github
For long sequence embedding or using LucaOne for downstream tasks, please refer to the git repository:

https://github.com/LucaOne/LucaOne,       
https://github.com/LucaOne/LucaOneTaks