import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", default='Qwen/Qwen2.5-VL-7B-Instruct', type=str)

args = parser.parse_args()

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)   

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
        # Convert to float32 before SVD
        A_float = A.to(torch.float32)
        eig_val = torch.svd(A_float / torch.trace(A_float))[1] 
        entropy = - (eig_val * torch.log(eig_val)).nansum().item()
        erank = math.exp(entropy)
    return erank

def compute(R):
    return cal_erank(cal_cov(normalize(R)))

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.path, torch_dtype="auto", device_map="cuda"
).eval()

# default processer

processor = AutoProcessor.from_pretrained(args.path,max_pixels = 1204224)

def cal_erank_mm(img, inst, response, processor):
    # Prepare messages including assistant's response
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                },
                {"type": "text", "text": inst},  # Instruction
            ],
        },
        {
            "role": "assistant",
            "content": response
        }
    ]

    # Generate full text
    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)

    # Tokenize to get inputs
    inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    last_hidden_state = outputs.hidden_states[-1][0, :, :]  # Get the last hidden state
    R = normalize(last_hidden_state)
    A = cal_cov(R)
    erank = cal_erank(A)
    
    return erank

img = '../assets/example.jpg'
inst = 'In the given diagram, quadrilateral ABCD is inscribed in circle O. The diameter BE is part of circle O and AE is connected to it. If the measure of angle BCD is twice the measure of angle BAD, then what is the degree measure of angle DAE?'
r = '\\boxed{30^\circ}'

erank = cal_erank_mm(img, inst, f'Answer: {r}', processor)

print(erank)

