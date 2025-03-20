'''
import ollama
import requests
import bs4
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser

def scrape_webpage(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        for tag in soup(['script', 'style', 'img', 'table']):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        cleaned_text = re.sub(r'\n+', '\n', text)
        return cleaned_text
    else:
        raise Exception(f"Failed to fetch {url} - Status Code: {response.status_code}")

url = "https://docs.ultralytics.com/models/yolo12/"
web_content = scrape_webpage(url)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.create_documents([web_content])

embeddings = OllamaEmbeddings(model="nomic-embed-text")

persist_directory = "./vector_db"
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory
)
vectorstore.persist()

retriever = vectorstore.as_retriever()

def combine_docs(docs):
    return " ".join([re.sub(r'\n+', ' ', doc.page_content).strip() for doc in docs])

def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='deepseek-r1:8b', messages=[{'role': 'user', 'content': formatted_prompt}], stream=True)
    return response

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    print(formatted_context)
    return ollama_llm(question, formatted_context)

demo_question = "What is yolov12 ?"
result = rag_chain(demo_question)
print("Demo Question:", demo_question)
for chunk in result:
  print(chunk['message']['content'], end='', flush=True)
'''

import ollama
import trafilatura
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser

def scrape_webpage(url):
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise Exception(f"Failed to fetch {url}")
    extracted = trafilatura.extract(downloaded, include_comments=False, include_formatting=False)
    cleaned_text = re.sub(r'\n+', '\n', extracted).strip() if extracted else ""
    return cleaned_text

url = "https://qwenlm.github.io/blog/qwq-32b/"
web_content = scrape_webpage(url)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.create_documents([web_content])

embeddings = OllamaEmbeddings(model="nomic-embed-text")

persist_directory = "./vector_db"
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory
)
vectorstore.persist()

retriever = vectorstore.as_retriever()

def combine_docs(docs):
    return " ".join([re.sub(r'\n+', ' ', doc.page_content).strip() for doc in docs])

def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='deepseek-r1:8b', messages=[{'role': 'user', 'content': formatted_prompt}], stream=True)
    return response

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    print(formatted_context)
    return ollama_llm(question, formatted_context)

demo_question = "Who is usa president ?"
result = rag_chain(demo_question)
print("Demo Question:", demo_question)
for chunk in result:
    print(chunk['message']['content'], end='', flush=True)

