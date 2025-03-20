'''
import os
import ollama
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set a similarity threshold
SIMILARITY_THRESHOLD = 0.4

# Initialize ChromaDB with embeddings
embedding_function = OllamaEmbeddings(model='nomic-embed-text')
vectorstore = Chroma(persist_directory="./vector_db", embedding_function=embedding_function)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Function to list saved conversations
def list_saved_conversations():
    chats_dir = "chats"
    if not os.path.exists(chats_dir):
        os.makedirs(chats_dir)
    return [f for f in os.listdir(chats_dir) if f.endswith('.txt')]

# Function to load conversation history
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
            current_message += "\n" + line.strip()

    if current_message:
        conversation_history.append({"role": current_role, "content": current_message.strip()})
    
    return conversation_history

# Function to save conversation history
def save_conversation(conversation_history, file_name):
    chats_dir = "chats"
    if not os.path.exists(chats_dir):
        os.makedirs(chats_dir)
    file_path = os.path.join(chats_dir, f"{file_name}.txt")

    with open(file_path, 'w') as file:
        for message in conversation_history:
            file.write(f"{message['role']}: {message['content']}\n")

# Function to extract relevant documents
def retrieve_relevant_context(query):
    retrieved_docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
    filtered_docs = [doc for doc, score in retrieved_docs_with_scores if score > SIMILARITY_THRESHOLD]
    
    if not filtered_docs:
        return ""
    
    return " ".join([re.sub(r'\n+', ' ', doc.page_content).strip() for doc in filtered_docs])

# Function to interact with Ollama LLM
def ollama_llm(question, context):
    prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='deepseek-r1:8b', messages=[{'role': 'user', 'content': prompt}], stream=True)

    return response

# Main chatbot loop
def chat_with_bot():
    conversation_history = [{"role": "system", "content": "You are a helpful AI assistant."}]

    # Handle saved conversations
    saved_conversations = list_saved_conversations()
    if saved_conversations:
        print("Saved conversations:")
        for i, conversation in enumerate(saved_conversations):
            print(f"{i+1}. {conversation}")

        load_or_start_new = input("Do you want to load a saved conversation or start a new one? (load/new): ")
        if load_or_start_new.lower() == 'load':
            selected_index = int(input("Enter the number of the conversation you want to load: ")) - 1
            selected_conversation = saved_conversations[selected_index]
            conversation_history = load_conversation(selected_conversation)

    while True:
        input_string = ""
        while True:
            line = input("> ")
            if line == ",,,":
                break
            elif line.lower() == "quit":
                input_string += line
                break
            else:
                input_string += line + "\n"
        
        if input_string.lower() == "quit":
            break

        context = retrieve_relevant_context(input_string)
        response = ollama_llm(input_string, context)

        assistant_response = ""
        for chunk in response:
            assistant_response += chunk['message']['content']

        conversation_history.append({"role": "user", "content": input_string.strip()})
        conversation_history.append({"role": "assistant", "content": assistant_response.strip()})

        print(f"Assistant: {assistant_response}")

    # Save conversation session
    save_chat = input("Do you want to save this chat? (yes/no): ")
    if save_chat.lower() == 'yes':
        file_name = input("Enter a name for the conversation file: ")
        if file_name:
            save_conversation(conversation_history, file_name)

# Run the chatbot
chat_with_bot()

'''
#!/root/miniconda2/envs/aagent/bin/python

import os
import ollama
import re
import trafilatura
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Set a similarity threshold
SIMILARITY_THRESHOLD = 0.4

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

# URL to scrape (web-based)
url = "https://docs.ultralytics.com/models/yolo12/"
web_content = scrape_webpage(url)
print("Web content fetched.")

# Split the content into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.create_documents([web_content])
print(f"Created {len(splits)} document chunks.")

chroma_config = {
    'index_name': 'knowledge_base',
    'vector_search_params': {
        'limit': 5,
        'filter_threshold': 0.75,
        'include_metadata': True
    },
    'retrieval': {
        'knn': True,
        'exact_search': False
    }
}

# Initialize embeddings and vector store with persistent storage
embeddings = OllamaEmbeddings(model="nomic-embed-text")
persist_directory = "./vector_db"
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory,
    config=chroma_config
)


# Create a retriever from the vector store
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def combine_docs(docs):
    # Combine document texts into one context string
    return " ".join([re.sub(r'\n+', ' ', doc.page_content).strip() for doc in docs])

'''
def retrieve_relevant_context(query):
    # Use retriever's similarity search with score if available
    try:
        results = retriever.similarity_search_with_score(query, k=3)
    except Exception as e:
        print("Error during similarity_search_with_score:", e)
        # Fallback to simple invoke if necessary
        results = [(doc, 1.0) for doc in retriever.invoke(query)]
    
    # Filter based on threshold
    filtered_docs = [doc for doc, score in results if score > SIMILARITY_THRESHOLD]
    if not filtered_docs:
        return ""
    return combine_docs(filtered_docs)
'''
def retrieve_relevant_context(query):
    try:
        results = vectorstore.similarity_search(query, k=3)  # Retrieve without scores
    except Exception as e:
        print("Error during similarity search:", e)
        return ""

    if not results:
        return ""

    return combine_docs(results)

def ollama_llm(question, context):
    if not context:
        context = "No relevant information found in the knowledge base."
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
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
    context = retrieve_relevant_context(question)
    print("Retrieved context:", context)
    return ollama_llm(question, context)

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


    while True:
        user_input = ""
        while True:
            line = input("> ")
            if line == ",,,":
                break
            if line == "quit":
                user_input = "quit"
                break
            user_input += line + "\n"
        if user_input.lower() == "quit":
            break

        conversation_history.append({"role": "user", "content": user_input.strip()})
        response_iterator = rag_chain(user_input.strip())
        assistant_response = ""
        for chunk in response_iterator:
            assistant_response += chunk['message']['content']
            print(chunk['message']['content'], end='', flush=True)
        conversation_history.append({"role": "assistant", "content": assistant_response.strip()})

    save_opt = input("Save this conversation? (yes/no): ")
    if save_opt.lower() == "yes":
        file_name = input("Enter conversation file name: ").strip()
        if file_name:
            save_conversation(conversation_history, file_name)

chat_with_bot()

