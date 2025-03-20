#!/root/miniconda2/envs/aagent/bin/python

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
