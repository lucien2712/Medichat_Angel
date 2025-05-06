import os
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from peft import PeftModel
import argparse

parser = argparse.ArgumentParser(description='輸入模型設定資訊')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--peft_path', type=str, required=False)
parser.add_argument('--device_map', type=str, default="auto", required=False)


args = parser.parse_args()

model = args.model
tokenizer = args.model

device_map = args.device_map
if device_map == "zero":
        device_map = "balanced_low_0"

os.environ["CUDA_VISIBLE_DEVICES"]="0"

print("Loading "+model+"...")
gpu_count = torch.cuda.device_count()
print('gpu_count', gpu_count)


#------------------------------------------------------------>load model
tokenizer = LlamaTokenizer.from_pretrained(tokenizer)
model = LlamaForCausalLM.from_pretrained(
        model,
        device_map=device_map,
        torch_dtype=torch.float16,
        load_in_8bit=False,
        #cache_dir="cache"
    )
#------------------------------------------------------------>load model from lora weight
if args.peft_path:
    peft_path = args.peft_path
    model = PeftModel.from_pretrained(model, peft_path, torch_dtype=torch.float16)


#------------------------------------------------------------->conversation setting
First_chat = "醫療小天使: 我是醫療小天使，有什麼醫療問題想問我呢?"
print(First_chat)
history = []
history.append(First_chat)

def go():
    invitation = "醫療小天使: "
    human_invitation = "病人: "

    # input
    msg = input(human_invitation)
    print("")

    history.append(human_invitation + msg)

    fulltext = "如果你是醫生，請根據病人的描述以專業醫療知識回應。請注意描述中的重要醫學資訊，如身體部位的症狀、頻率、持續時間。 \n" + "\n".join(history) + "\n" + invitation
    
    generated_text = ""
    gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()

    with torch.no_grad():
            generated_ids = model.generate(
                input_ids=gen_in,
                temperature=0.2,
                top_k=40,
                top_p=0.9,
                do_sample=True,
                num_beams=1,
                repetition_penalty=1.3,
                max_new_tokens=200
            )
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            text_without_prompt = generated_text[len(fulltext):]

    response = text_without_prompt

    response = response.split(human_invitation)[0]

    response.strip()

    print(invitation + response)

    print("")

    history.append(invitation + response)

while True:
    go()

