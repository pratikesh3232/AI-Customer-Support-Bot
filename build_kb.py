# build_kb.py
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

loaders = [
    TextLoader("docs/faq.txt"),
    TextLoader("docs/return_policy.txt"),
    PyPDFLoader("docs/product_manual.pdf"),
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_kb")

print(f"Done — {len(chunks)} chunks saved.")