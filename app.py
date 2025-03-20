'''
import ollama
import requests
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser

# Function to scrape webpage content using requests & BeautifulSoup
def scrape_webpage(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        return soup.get_text()  # Extracting raw text from HTML
    else:
        raise Exception(f"Failed to fetch {url} - Status Code: {response.status_code}")

# Define the website URL to scrape
url = "https://docs.ultralytics.com/models/yolo12/"
web_content = scrape_webpage(url)
print(web_content)
# 1. Split the extracted text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.create_documents([web_content])

# 2. Generate embeddings using nomic-embed-text via Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 3. Set up ChromaDB with persistent storage
persist_directory = "./vector_db"
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory
)
vectorstore.persist()  # Ensure persistence for future additions

# 4. Function to query the RAG system
retriever = vectorstore.as_retriever()

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='deepseek-r1:8b', messages=[{'role': 'user', 'content': formatted_prompt}], stream=True)
    return response

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
#    print(formatted_context)
    return ollama_llm(question, formatted_context)

# 5. Test the pipeline with a demo query
demo_question = "What is yolov12 ?"
result = rag_chain(demo_question)
print("Demo Question:", demo_question)
#print("Answer:", result)
for chunk in result:
  print(chunk['message']['content'], end='', flush=True)

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
        for script in soup(["script", "style", "img", "table"]):
            script.extract()
        text = soup.get_text()
        text = re.sub(r'\n+', '\n', text).strip()
        return text
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
    return "\n\n".join(doc.page_content for doc in docs)

def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='deepseek-r1:8b', messages=[{'role': 'user', 'content': formatted_prompt}], stream=True)
    return response

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    print(formatted_context)
    return ollama_llm(question, formatted_context)

demo_question = "What is YOLOv12?"
result = rag_chain(demo_question)
print("Demo Question:", demo_question)
for chunk in result:
    print(chunk['message']['content'], end='', flush=True)

