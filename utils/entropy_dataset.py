from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Model, AutoModel
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
        #entropy = - (eig_val * torch.log(eig_val)).nansum().cpu().item()   
        entropy = - (eig_val * torch.log(eig_val)).nansum().item()
        normalized_entropy = entropy/math.log(A.shape[0])
        normalized_entropy1 = math.exp(entropy)/A.shape[0]
    return normalized_entropy, normalized_entropy1

def jsonl_to_list(filename):
    data_list = []

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            data_list.append(data['chosen'])

    return data_list

def main(args):
    if args.model == "gpt":
        model_path = f"cerebras/Cerebras-GPT-{args.size}" # 111M 256M 590M 1.3B 2.7B 6.7B 13B
    elif args.model == "pythia":
        model_path = f"EleutherAI/pythia-{args.size}" # 14m 70m 160m 410m 1b 1.4b 2.8b 6.9b 12b
    else:
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, device_map="auto")
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

    
    ls1, ls2, ls3 = [], [], []
    with tqdm.tqdm(input_ids, desc="Entropy: - ") as progress:
        for id in progress:
            with torch.no_grad():
                r = model(id.cuda())[0][0, :, :]
                R = normalize(r)
                A = cal_cov(R)
                Entropy1, Entropy3 = cal_entropy(A)
                ls1.append(Entropy1)
                ls3.append(Entropy3)
            torch.cuda.empty_cache()
            progress.set_description(f"Entropy: {Entropy1:.4f}")

    entropy_avg1 = sum(ls1) / len(ls1)
    print(A.shape[0])
    dim = A.shape[0]
    entropy_avg2 = math.exp(entropy_avg1 * math.log(dim)) / dim
    entropy_avg3 = sum(ls3) / len(ls3)
    print(f'alg a: {entropy_avg1}')
    print(f'alg b: {entropy_avg2}')
    print(f'alg c: {entropy_avg3}')
 

 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--size", type=str, default="111M")
    parser.add_argument("--dataset", type=str, default="dolly")
    args = parser.parse_args()
    
    main(args)