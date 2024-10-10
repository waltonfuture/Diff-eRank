from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import math
import json
import tqdm
import random
from datasets import load_dataset, load_from_disk

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
    return entropy

def compute(R):
    return cal_entropy(cal_cov(normalize(R)))

def jsonl_to_list(filename):
    data_list = []

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            data_list.append(data['chosen'])

    return data_list

def main(args):
    model_path = "facebook/opt-1.3b" # for example
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).cuda()
    config = AutoConfig.from_pretrained(model_path)
    untrained_model = AutoModel.from_config(config).to('cuda')
    input_ids = []

    if args.dataset == "dolly":
        with open('/path/to/datasets/databricks-dolly-15k/databricks-dolly-15k.jsonl', 'r') as file:
            for line in file:
                json_line = json.loads(line)
                context = json_line.get('context', '')  
                
                if len(context)>0:
                    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)                
                    input_ids.append(inputs.input_ids)

    elif args.dataset == "wiki":
        dataset = load_from_disk("/path/to/datasets/wiki")
        sample_size = 10000
        random_seed = 123
        random.seed(random_seed)
        dataset_size = len(dataset['train'])
        random_indices = random.sample(range(dataset_size), sample_size)

        for i, idx in tqdm.tqdm(enumerate(random_indices)):
            sample = dataset['train'][idx]
            context = sample['text']
            inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)
                     
            input_ids.append(inputs.input_ids)

    elif args.dataset == "rlhf":
        ref_list1 = jsonl_to_list('/path/to/datasets/hh-rlhf/harmless-base/test.jsonl')
        ref_list2 = jsonl_to_list('/path/to/datasets/hh-rlhf/helpful-base/test.jsonl')
        ref_list3 = jsonl_to_list('/path/to/datasets/hh-rlhf/helpful-online/test.jsonl')
        ref_list4 = jsonl_to_list('/path/to/datasets/hh-rlhf/helpful-rejection-sampled/test.jsonl')
        dataset = ref_list1 + ref_list2 + ref_list3 + ref_list4
        for context in tqdm.tqdm(dataset):
            inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)              
            input_ids.append(inputs.input_ids)
    
    elif args.dataset == "openwebtext2":
        dataset = load_from_disk("/path/to/datasets/pile_openwebtext2")['validation']
        sample_size = 10000
        random_seed = 123
        random.seed(random_seed)
        dataset_size = len(dataset)
        random_indices = random.sample(range(dataset_size), sample_size)
        for i, idx in enumerate(random_indices):
            context = dataset[idx]['text']
            inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048).to('cuda')            
            input_ids.append(inputs)

    
    ls1, ls2 = [], []
    with tqdm.tqdm(input_ids) as progress:
        for id in progress:
            with torch.no_grad():
                R1 = model(id.cuda())[0][0, :, :]
                entropy1 = compute(R1)
                R2 = untrained_model(id.cuda())[0][0, :, :]
                entropy2 = compute(R2)
                ls1.append(entropy1)
                ls2.append(entropy2)
    erank1 = math.exp(sum(ls1) / len(ls1))
    erank2 = math.exp(sum(ls2) / len(ls2))
    print(erank2 - erank1) 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Diff-eRank of dataset')
    parser.add_argument("--dataset", type=str, default="dolly")
    args = parser.parse_args()
    
    main(args)