import os
import ollama
import re
import trafilatura
import base64
from typing import Dict, Callable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Boolean flag to switch RAG on/off.
USE_RAG = False

# Similarity threshold for context retrieval.
SIMILARITY_THRESHOLD = 0.4

def scrape_webpage(url):
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise Exception(f"Failed to fetch {url}")
    extracted = trafilatura.extract(downloaded, include_comments=False, include_formatting=False)
    if not extracted:
        raise Exception("Failed to extract text from the page.")
    cleaned_text = re.sub(r'\n+', '\n', extracted).strip()
    return cleaned_text

# Web-based URL to scrape.
url = "https://docs.ultralytics.com/models/yolo12/"
web_content = scrape_webpage(url)

# Split content into chunks.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.create_documents([web_content])

# Initialize embeddings and vector store with persistent storage.
embeddings = OllamaEmbeddings(model="nomic-embed-text")
persist_directory = "./vector_db"
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory
)

# Create a retriever from the vector store.
def combine_docs(docs):
    return " ".join([re.sub(r'\n+', ' ', doc.page_content).strip() for doc in docs])

def retrieve_relevant_context(query):
    results = vectorstore.similarity_search_with_score(query, k=3)
    filtered_docs = [doc for doc, score in results if score > SIMILARITY_THRESHOLD]
    if not filtered_docs:
        return ""
    return combine_docs(filtered_docs)

def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    print(context)
    response = ollama.chat(
        model='llama3.1:8b',
        messages=[{'role': 'user', 'content': formatted_prompt}],
        tools=[call_llava_tool],
        stream=True
    )
    return response

# Function calling for image processing.
def call_llava(image_path, question):
  #  with open(image_path, "rb") as f:
 #       b64_img = base64.b64encode(f.read()).decode("utf-8")
#    prompt = f"Image (base64): {b64_img}\nQuestion: {question}"
#    print(prompt)
    response = ollama.chat(
        model='llava:13b',
        messages=[{'role': 'user', 'content': question, 'images': [image_path]}],
        stream=True
    )
    return response

# Manual tool definition for image processing.
call_llava_tool = {
    'type': 'function',
    'function': {
        'name': 'call_llava',
        'description': 'Process an image using LLaVA',
        'parameters': {
            'type': 'object',
            'required': ['image_path', 'question'],
            'properties': {
                'image_path': {'type': 'string', 'description': 'Path to the image'},
                'question': {'type': 'string', 'description': 'Question about the image'},
            },
        },
    },
}

# Available function mapping.
available_functions: Dict[str, Callable] = {
    'call_llava': call_llava,
}

def rag_chain(question):
    context = retrieve_relevant_context(question) if USE_RAG else ""
    response = ollama_llm(question, context)

    if response.message.tool_calls:
        for tool in response.message.tool_calls:
            if function_to_call := available_functions.get(tool.function.name):
                return function_to_call(**tool.function.arguments)

    return response['message']['content']

def chat_with_bot():
    while True:
        user_input = input("> ")
        if user_input.lower() == "quit":
            break
        response = rag_chain(user_input.strip())
#        print(response)
        for chunk in response:
            assistant_response += chunk['message']['content']
            print(chunk['message']['content'], end='', flush=True)

chat_with_bot()
