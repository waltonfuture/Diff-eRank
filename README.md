# Diff-eRank: A Novel Rank-Based Metric for Evaluating Large Language Models (NeurIPS 2024)
[Lai Wei](https://waltonfuture.github.io/) *, Zhiquan Tan *, Chenghai Li, [Jindong Wang](https://jd92.wang/), [Weiran Huang](https://www.weiranhuang.com/) (*Equal Contribution).

**Shanghai Jiao Tong University & Tsinghua University & Microsoft Research Asia**

<a href='https://arxiv.org/abs/2401.17139'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://zhuanlan.zhihu.com/p/687278237'><img src='https://img.shields.io/badge/Project-Link-Green'></a>


## Introduction
We introduce a rank-based metric called Diff-eRank, which is rooted in information theory and geometry principles. Diff-eRank evaluates LLMs by examining their hidden representations to quantify how LLMs discard redundant information after training.
Specifically, we demonstrate its applicability in both single-modal (language) and multi-modal settings. For language models, our findings reveal that the Diff-eRank increases when the model scales up, which also demonstrates a consistent relationship with traditional metrics like loss and accuracy.
For multi-modal models, we also propose an evaluation method based on rank for assessing alignment quality and we find that modern multi-modal large language models exhibit good alignment performance. 

## Calculation of Diff-eRank

### Setup
```bash
pip install transformers torch datasets
```

### Calculation

```bash
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import math

# R input N*d
def normalize(R):
    with torch.no_grad():
        mean = R.mean(dim=0)
        R = R - mean
        norms = torch.norm(R, p=2, dim=1, keepdim=True)
        R = R/norms
    return R

def cal_cov(R):
    with torch.no_grad():
        Z = torch.nn.functional.normalize(R, dim=1)
        A = torch.matmul(Z.T, Z)/Z.shape[0]
    return A

def cal_erank(A):
    with torch.no_grad():
        eig_val = torch.svd(A / torch.trace(A))[1] 
        entropy = - (eig_val * torch.log(eig_val)).nansum().item()
        erank = math.exp(entropy)
    return erank

def compute(R):
    return cal_erank(cal_cov(normalize(R)))

model_path = "facebook/opt-1.3b" # for example
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).cuda()
config = AutoConfig.from_pretrained(model_path)
untrained_model = AutoModel.from_config(config).to('cuda')

text = "We introduce a rank-based metric called Diff-eRank, which is rooted in information theory and geometry principles. Diff-eRank evaluates LLMs by examining their hidden representations to quantify how LLMs discard redundant information after training." # for example
inputs = tokenizer(text, return_tensors="pt").to('cuda')
with torch.no_grad():
    R1 = model(inputs.input_ids)[0][0, :, :]
    R2 = untrained_model(inputs.input_ids)[0][0, :, :]
    erank1 = compute(R1)
    erank2 = compute(R2)
    RD = erank2 - erank1
print(RD)
```
### Diff-eRank of Single Sentence
```
cd utils

python diff_erank_single_sentence.py
```

### Diff-eRank of Dataset

Please download the datasets of [wiki-en](https://huggingface.co/datasets/wikipedia), [dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), [openwebtext2](https://huggingface.co/datasets/suolyer/pile_openwebtext2), [hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) in huggingface and edit the data path in your scripts.

```
cd utils

python diff_erank_dataset.py
```

## Citation

If you're using Diff-eRank in your research or applications, please cite using this BibTeX:
```bibtex
@article{wei2024large,
  title={Large Language Model Evaluation via Matrix Entropy},
  author={Wei, Lai and Tan, Zhiquan and Li, Chenghai and Wang, Jindong and Huang, Weiran},
  journal={arXiv preprint arXiv:2401.17139},
  year={2024}
}
```
