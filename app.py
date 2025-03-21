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

'''

import ollama
import trafilatura
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Set a similarity threshold
SIMILARITY_THRESHOLD = 0.6  # Adjust based on testing

def scrape_webpage(url):
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise Exception(f"Failed to fetch {url}")
    extracted = trafilatura.extract(downloaded, include_comments=False, include_formatting=False)
    cleaned_text = re.sub(r'\n+', '\n', extracted).strip() if extracted else ""
    return cleaned_text

url = "https://docs.ultralytics.com/models/yolo12/"
web_content = scrape_webpage(url)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
splits = text_splitter.create_documents([web_content])

embeddings = OllamaEmbeddings(model="nomic-embed-text")

persist_directory = "./vector_db"
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory
)
vectorstore.persist()

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def combine_docs(docs):
    return " ".join([re.sub(r'\n+', ' ', doc.page_content).strip() for doc in docs])

def ollama_llm(question, context):
    if not context:
        context = " "
    print(context)
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='deepseek-r1:8b', messages=[{'role': 'user', 'content': formatted_prompt}], stream=True)
    return response

def rag_chain(question):
    retrieved_docs_with_scores = vectorstore.similarity_search_with_score(question, k=3)
    # Filter relevant documents based on similarity threshold
    filtered_docs = [doc for doc, score in retrieved_docs_with_scores if score > SIMILARITY_THRESHOLD]
    formatted_context = combine_docs(filtered_docs) if filtered_docs else ""
    return ollama_llm(question, formatted_context)

# Test the pipeline with a related and unrelated query
demo_question = "What are black holes ?"
result = rag_chain(demo_question)
print("\nDemo Question:", demo_question)
for chunk in result:
    print(chunk['message']['content'], end='', flush=True)
