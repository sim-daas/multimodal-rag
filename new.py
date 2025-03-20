#!/root/miniconda2/envs/aagent/bin/python

import os
import ollama
import re
import trafilatura
import requests
import json
import time
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Set a similarity threshold
SIMILARITY_THRESHOLD = 0.4
# Default setting for web search (can be toggled by user)
WEB_SEARCH_ENABLED = True
embeddings = OllamaEmbeddings(model="nomic-embed-text")

persist_directory = "./vector_db"
vectorstore = Chroma(
#    documents=splits,
    embedding_function=embeddings,
    persist_directory=persist_directory
)

vectorstore.vector_search_params = {
    'limit': 5,
    'filter_threshold': 0.75,
    'include_metadata': True
}

vectorstore.retrieval = {
    'knn': False,
    'exact_search': False
}


# --- Web Search Functions ---
def search_duckduckgo(query, num_results=3):
    """Search DuckDuckGo and return results."""
    try:
        # Prepare the search URL with the query
        search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the HTML response
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='result')
        
        search_results = []
        for i, result in enumerate(results):
            if i >= num_results:
                break
                
            title_elem = result.find('a', class_='result__a')
            snippet_elem = result.find('a', class_='result__snippet')
            
            if title_elem and snippet_elem:
                title = title_elem.get_text()
                snippet = snippet_elem.get_text()
                url = title_elem.get('href')
                
                # Extract actual URL from DuckDuckGo redirect URL
                if url.startswith('/'):
                    url_params = url.split('uddg=')
                    if len(url_params) > 1:
                        url = requests.utils.unquote(url_params[1].split('&')[0])
                
                search_results.append({
                    'title': title,
                    'snippet': snippet,
                    'url': url
                })
        
        return search_results
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []

def extract_content_from_url(url):
    """Extract content from a URL using trafilatura."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return f"Failed to fetch content from {url}"
            
        extracted = trafilatura.extract(downloaded, include_comments=False, include_formatting=False)
        if not extracted:
            return f"No extractable content from {url}"
            
        # Clean and limit the content
        cleaned_text = re.sub(r'\n+', '\n', extracted).strip()
        # Limit to approximately 2000 characters to avoid overloading the context
        if len(cleaned_text) > 2000:
            cleaned_text = cleaned_text[:2000] + "..."
            
        return cleaned_text
    except Exception as e:
        return f"Error extracting content: {str(e)}"

def should_search_web(query, local_context):
    """Determine if web search should be triggered."""
    # Always search if no local context found
    if not local_context or len(local_context.strip()) < 50:
        return True
        
    # Check for search intent keywords
    search_keywords = ["latest", "current", "recent", "news", "update", "today", 
                      "yesterday", "last week", "this year", "2023", "2024"]
    
    # Look for time-sensitive queries
    for keyword in search_keywords:
        if keyword.lower() in query.lower():
            return True
    
    # If local context seems insufficient
    if len(local_context.strip().split()) < 30:
        return True
        
    return False

def get_web_context(query):
    """Search the web and compile context from search results."""
    search_results = search_duckduckgo(query)
    if not search_results:
        return "No search results found."
    
    web_context = f"Web Search Results for: {query}\n\n"
    
    # First add all snippets for quick context
    for i, result in enumerate(search_results):
        web_context += f"{i+1}. {result['title']}: {result['snippet']}\n"
    
    web_context += "\n\nDetailed content from top result:\n"
    
    # Get detailed content from the top result
    if search_results and 'url' in search_results[0]:
        content = extract_content_from_url(search_results[0]['url'])
        web_context += f"Source: {search_results[0]['url']}\n{content}\n"
    
    return web_context

def combine_docs(docs):
    # Combine document texts into one context string
    return " ".join([re.sub(r'\n+', ' ', doc.page_content).strip() for doc in docs])


# --- Existing functions with web search integration ---
def scrape_webpage(url):
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise Exception(f"Failed to fetch {url}")
    extracted = trafilatura.extract(downloaded, include_comments=False, include_formatting=False)
    if not extracted:
        raise Exception("Failed to extract text from the page.")
    # Remove extraneous whitespace
    cleaned_text = re.sub(r'\n+', '\n', extracted).strip()
    return cleaned_text

def retrieve_relevant_context(query):
    try:
        results = vectorstore.similarity_search(query, k=3)  # Retrieve without scores
    except Exception as e:
        print("Error during similarity search:", e)
        return ""

    if not results:
        return ""

    return combine_docs(results)

def ollama_llm(question, context, web_context=""):
    if not context:
        if web_context:
            context = f"No relevant information found in local knowledge base, but web search provided: {web_context}"
        else:
            context = "No relevant information found in the knowledge base."
    elif web_context:
        context = f"From local knowledge base: {context}\n\nFrom web search: {web_context}"
        
    formatted_prompt = f"""Question: {question}

Context: {context}

Answer the question based on the provided context. If the information is from web search, 
mention that in your response. If you don't know the answer, say so clearly."""

    try:
        response = ollama.chat(
            model='deepseek-r1:8b',
            messages=[{'role': 'user', 'content': formatted_prompt}],
            stream=True
        )
    except Exception as e:
        print("Error during Ollama chat:", e)
        return iter([{'message': {'content': "Error contacting LLM."}}])
    return response

def rag_chain(question):
    # Get context from local knowledge base
    local_context = retrieve_relevant_context(question)
    print("Retrieved local context:", local_context[:100] + "..." if len(local_context) > 100 else local_context)
    
    # Determine if web search is needed
    web_context = ""
    if WEB_SEARCH_ENABLED and should_search_web(question, local_context):
        print("Searching the web for additional information...")
        web_context = get_web_context(question)
        print("Retrieved web context:", web_context[:100] + "..." if len(web_context) > 100 else web_context)
    
    # Generate response using both contexts
    return ollama_llm(question, local_context, web_context)

# --- Conversation saving/loading functions ---
def list_saved_conversations():
    chats_dir = "chats"
    if not os.path.exists(chats_dir):
        os.makedirs(chats_dir)
    return [f for f in os.listdir(chats_dir) if f.endswith('.txt')]

def load_conversation(file_name):
    chats_dir = "chats"
    file_path = os.path.join(chats_dir, file_name)
    conversation_history = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    current_role, current_message = "", ""
    for line in lines:
        if line.startswith("user:") or line.startswith("assistant:"):
            if current_message:
                conversation_history.append({"role": current_role, "content": current_message.strip()})
            current_role = line.split(":")[0]
            current_message = line.split(":", 1)[1].strip()
        else:
            current_message += " " + line.strip()
    if current_message:
        conversation_history.append({"role": current_role, "content": current_message.strip()})
    return conversation_history

def save_conversation(conversation_history, file_name):
    chats_dir = "chats"
    if not os.path.exists(chats_dir):
        os.makedirs(chats_dir)
    file_path = os.path.join(chats_dir, f"{file_name}.txt")
    with open(file_path, 'w') as file:
        for message in conversation_history:
            file.write(f"{message['role']}: {message['content']}\n")

# --- Main chatbot loop ---
def chat_with_bot():
    global WEB_SEARCH_ENABLED
    conversation_history = [{"role": "system", "content": "You are a helpful AI assistant."}]
    
    saved_conversations = list_saved_conversations()
    if saved_conversations:
        print("Saved conversations:")
        for i, conv in enumerate(saved_conversations):
            print(f"{i+1}. {conv}")
        option = input("Load saved conversation or start new? (load/new): ")
        if option.lower() == "load":
            idx = int(input("Enter the number to load: ")) - 1
            conversation_history = load_conversation(saved_conversations[idx])
        else:
            print("Starting new conversation.")
    
    # Ask user if they want web search enabled
    search_option = input("Enable web search capability? (yes/no, default: yes): ").lower()
    WEB_SEARCH_ENABLED = search_option != "no"
    print(f"Web search is {'enabled' if WEB_SEARCH_ENABLED else 'disabled'}")

    while True:
        user_input = ""
        while True:
            line = input("> ")
            if line == ",,,":
                break
            if line == "quit":
                user_input = "quit"
                break
            if line == "!websearch on":
                WEB_SEARCH_ENABLED = True
                print("Web search enabled")
                user_input = ""
                break
            if line == "!websearch off":
                WEB_SEARCH_ENABLED = False
                print("Web search disabled")
                user_input = ""
                break
            user_input += line + "\n"
        
        if not user_input:
            continue
            
        if user_input.lower() == "quit":
            break

        conversation_history.append({"role": "user", "content": user_input.strip()})
        print("\nProcessing...", end="\r")
        response_iterator = rag_chain(user_input.strip())
        print(" " * 12, end="\r")  # Clear "Processing..." message
        
        assistant_response = ""
        for chunk in response_iterator:
            assistant_response += chunk['message']['content']
            print(chunk['message']['content'], end='', flush=True)
        print("\n")  # Add a newline after response
        conversation_history.append({"role": "assistant", "content": assistant_response.strip()})

    save_opt = input("Save this conversation? (yes/no): ")
    if save_opt.lower() == "yes":
        file_name = input("Enter conversation file name: ").strip()
        if file_name:
            save_conversation(conversation_history, file_name)

chat_with_bot()

