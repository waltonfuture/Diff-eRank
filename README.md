# Large Language Model Evaluation via Matrix Entropy
[Lai Wei](https://waltonfuture.github.io/) *, Zhiquan Tan *, Chenghai Li, [Jindong Wang](https://jd92.wang/), [Weiran Huang](https://www.weiranhuang.com/) (*Equal Contribution).

**Shanghai Jiao Tong University & Tsinghua University & Microsoft Research Asia**

<a href='https://arxiv.org/abs/2401.17139'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 


## Introduction
We introduce matrix entropy, a novel metric rooted in information theory and geometry principles to quantify the data compression proficiency in LLMs. It reflects the model's ability to extract relevant information and eliminate unnecessary elements, thereby providing insight into the language model's intrinsic capability. 
Specifically, we demonstrate its applicability in both single-modal (language) and multi-modal settings. For language models, our findings reveal that the matrix entropy of representations follows a scaling law type reduction when the model scales up, serving as a complement to the traditional loss scaling law. For the multi-modal setting, we also propose an evaluation method based on matrix entropy for assessing alignment quality and we find that modern large multi-modal models exhibit great alignment performance. 


## Calculation of Matrix Entropy
```bash
from transformers import AutoTokenizer, AutoModel
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

def cal_entropy(A):
    with torch.no_grad():
        eig_val = torch.svd(A / torch.trace(A))[1] 
        entropy = - (eig_val * torch.log(eig_val)).nansum().item()
        normalized_entropy = entropy/math.log(A.shape[0])
    return normalized_entropy

model_path = "cerebras/Cerebras-GPT-1.3B" # for example
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, device_map="auto").cuda()

text = "I love Generative AI very much." # for example
inputs = tokenizer(text, return_tensors="pt").to('cuda')
with torch.no_grad():
    R = model(inputs.input_ids)[0][0, :, :]
    R = normalize(R)
    A = cal_cov(R)
    Entropy = cal_entropy(A)
print(Entropy)
```
### Matrix Entropy of Single Sentence
```
cd utils

python entropy_single_sentence.py
```

### Matrix Entropy of Dataset

Please download the datasets of [wiki-en](https://huggingface.co/datasets/wikipedia), [dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), [openwebtext2](https://huggingface.co/datasets/suolyer/pile_openwebtext2), [hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) in huggingface and edit the data path in your scripts.

```
cd utils

python entropy_dataset.py
```

## Citation



If you're using Matrix Entropy in your research or applications, please cite using this BibTeX:
```bibtex
@misc{wei2024large,
      title={Large Language Model Evaluation via Matrix Entropy}, 
      author={Lai Wei and Zhiquan Tan and Chenghai Li and Jindong Wang and Weiran Huang},
      year={2024},
      eprint={2401.17139},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
