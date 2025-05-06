import os
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from peft import PeftModel
import argparse

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import SystemMessagePromptTemplate,ChatPromptTemplate

import torch
parser = argparse.ArgumentParser(description='輸入模型設定資訊')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--peft_path', type=str, required=False)
parser.add_argument('--device_map', type=str, default="auto", required=False)

args = parser.parse_args()

model = args.model
tokenizer = args.model

device_map = args.device_map
device = "gpu"
if device_map == "zero":
        device_map = "balanced_low_0"
        device = "cpu"

os.environ["CUDA_VISIBLE_DEVICES"]="0"

print("Loading "+model+"...")
gpu_count = torch.cuda.device_count()
print('gpu_count', gpu_count)


#embedding the document
embedding_model= tokenizer
embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                                model_kwargs={'device': device})   

loader = DirectoryLoader('./disease/', glob='*.txt',show_progress=True)
documents = loader.load()

# 初始化加载器
text_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=10)
# chunk_size參數用來指定每塊文本的大小(字元)、chunk_overlap參數用來指定每塊文本的重疊率。

# 切割加载的 document
split_docs = text_splitter.split_documents(documents)
docsearch = Chroma.from_documents(split_docs, embeddings)

#加載模型
#------------------------------------------------------------>load model
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
model.config.top_k = 40,
model.config.num_beams=1,
model.config.repetition_penalty=1.3,
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