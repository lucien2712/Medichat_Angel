from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

#embedding the document
embedding_model='GanymedeNil/text2vec-large-chinese'
embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                                model_kwargs={'device': device})   

loader = DirectoryLoader('./disease/', glob='*.txt',show_progress=True)

documents = loader.load()


text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)

split_docs = text_splitter.split_documents(documents)

# embedding and save
persist_directory = './db'
docsearch = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_directory)
docsearch.persist()