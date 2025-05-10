import os
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from peft import PeftModel
import argparse

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import SystemMessagePromptTemplate,ChatPromptTemplate

parser = argparse.ArgumentParser(description='輸入模型設定資訊')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--peft_path', type=str, required=False)
parser.add_argument('--device_map', type=str, default="auto", required=False)

args = parser.parse_args()

model = args.model
tokenizer = args.model

device_map = args.device_map
device = "cuda" if torch.cuda.is_available() else "cpu"
if device_map == "zero":
        device_map = "balanced_low_0"
        device = "cpu"

os.environ["CUDA_VISIBLE_DEVICES"]="0"

print("Loading "+model+"...")
gpu_count = torch.cuda.device_count()
print('gpu_count', gpu_count)


#embedding the document
embeddings = HuggingFaceEmbeddings(model_name='GanymedeNil/text2vec-large-chinese',
                                                model_kwargs={'device': device})   

# 從持久化目錄載入已建立的向量資料庫
persist_directory = './db'
docsearch = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

#加載模型
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

model.config.temperature = 0.2
model.config.top_p = 0.9
model.config.top_k = 40
model.config.num_beams=1
model.config.repetition_penalty=1.3
model.config.max_new_tokens=200


system_template = """如果你是醫生，請根據病人的描述以專業醫療知識回應。請注意描述中的重要醫學資訊，如身體部位的症狀、頻率、持續時間。\n\n 
                    {question}
                    \n\n
                    {chat_history}
                    """
messages = [
  SystemMessagePromptTemplate.from_template(system_template),
]
prompt = ChatPromptTemplate.from_messages(messages)

retriever = docsearch.as_retriever(search_kwargs={"k": 2})
#top_k(k): The number of documents to be ranked for each query.
#alpha: controls the relative importance of the document score and the document embedding distance in the ranking function.
#beta: controls the influence of the document score in the ranking function.

qa = ConversationalRetrievalChain.from_llm(llm=model, chain_type="hierarchical", retriever=retriever, 
                                           combine_docs_chain_kwargs={'prompt': prompt},return_source_documents= False)





First_chat = "醫療小天使: 我是醫療小天使，有什麼醫療問題想問我呢?"
print(First_chat)

history = [] #[(question, res["answer"]), ]
def go():
    invitation = "醫療小天使: "
    human_invitation = "病人: "

    # input
    question = input(human_invitation)
    print('')

    res = qa({"question":question, "chat_history":history})
    answer = res["answer"]
    print(invitation + answer)
    print()
    history.append((question, answer))

while True:
    go()