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