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