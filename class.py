#!/usr/bin/env python

import os
import ollama
import re
import trafilatura
import requests
import json
import time
import logging
from typing import List, Dict, Any, Optional, Iterator, Tuple
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ragchatbot")

class VectorStore:
    """Manages document embeddings and retrieval from local vector database."""
    
    def __init__(self, persist_directory: str = "./vector_db", model_name: str = "nomic-embed-text"):
        """Initialize the vector store with embedding model."""
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.embedding_function = None
        self.vectorstore = None
        self.similarity_threshold = 0.4
        self._initialize()
    
    def _initialize(self):
        """Initialize the embedding function and vector store."""
        try:
            self.embedding_function = OllamaEmbeddings(model=self.model_name)
            # Check if vector store exists
            if os.path.exists(self.persist_directory):
                logger.info(f"Loading existing vector store from {self.persist_directory}")
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_function
                )
            else:
                logger.warning(f"No existing vector store found at {self.persist_directory}. Creating empty store.")
                self.vectorstore = Chroma(
                    embedding_function=self.embedding_function,
                    persist_directory=self.persist_directory
                )
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def retrieve_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant context from vector store."""
        if not self.vectorstore:
            logger.warning("Vector store not initialized")
            return ""

        try:
            results = self.vectorstore.similarity_search(query, k=k)
            if not results:
                return ""

            return self._combine_docs(results)
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            return ""

    def _combine_docs(self, docs) -> str:
        """Combine document texts into one context string."""
        return " ".join([re.sub(r'\n+', ' ', doc.page_content).strip() for doc in docs])

    def add_document(self, document: str, source: str = ""):
        """Add a document to the vector store."""
        if not document:
            return

        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.create_documents([document], metadatas=[{"source": source}])
            self.vectorstore.add_documents(splits)
            logger.info(f"Added {len(splits)} document chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")

    def set_similarity_threshold(self, threshold: float):
        """Set the similarity threshold for retrieval."""
        self.similarity_threshold = threshold


class WebSearch:
    """Handles web search functionality and content extraction."""

    def __init__(self):
        """Initialize the web search component."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def search_duckduckgo(self, query: str, num_results: int = 3) -> List[Dict]:
        """Search DuckDuckGo and return results."""
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"

            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()

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
            logger.error(f"Search error: {str(e)}")
            return []

    def extract_content(self, url: str, max_chars: int = 2000) -> str:
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
            # Limit to approximately max_chars to avoid overloading the context
            if len(cleaned_text) > max_chars:
                cleaned_text = cleaned_text[:max_chars] + "..."

            return cleaned_text
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            return f"Error extracting content: {str(e)}"

    def get_web_context(self, query: str) -> str:
        """Search the web and compile context from search results."""
        search_results = self.search_duckduckgo(query)
        if not search_results:
            return "No search results found."

        web_context = f"Web Search Results for: {query}\n\n"
        
        # First add all snippets for quick context
        for i, result in enumerate(search_results):
            web_context += f"{i+1}. {result['title']}: {result['snippet']}\n"
        
        web_context += "\n\nDetailed content from top result:\n"
        
        # Get detailed content from the top result
        if search_results and 'url' in search_results[0]:
            content = self.extract_content(search_results[0]['url'])
            web_context += f"Source: {search_results[0]['url']}\n{content}\n"

        return web_context


class ConversationManager:
    """Handles saving and loading conversation history."""

    def __init__(self, chats_directory: str = "chats"):
        """Initialize the conversation manager."""
        self.chats_directory = chats_directory
        if not os.path.exists(self.chats_directory):
            os.makedirs(self.chats_directory)

    def list_conversations(self) -> List[str]:
        """List all saved conversations."""
        return [f for f in os.listdir(self.chats_directory) if f.endswith('.txt')]

    def load_conversation(self, file_name: str) -> List[Dict[str, str]]:
        """Load a conversation from a file."""
        file_path = os.path.join(self.chats_directory, file_name)
        conversation_history = []

        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            current_role, current_message = "", ""

            for line in lines:
                if line.startswith("user:") or line.startswith("assistant:") or line.startswith("system:"):
                    if current_message:
                        conversation_history.append({"role": current_role, "content": current_message.strip()})
                    current_role = line.split(":")[0]
                    current_message = line.split(":", 1)[1].strip()
                else:
                    current_message += " " + line.strip()

            if current_message:
                conversation_history.append({"role": current_role, "content": current_message.strip()})

            return conversation_history
        except Exception as e:
            logger.error(f"Error loading conversation: {str(e)}")
            return [{"role": "system", "content": "You are a helpful AI assistant."}]

    def save_conversation(self, conversation_history: List[Dict[str, str]], file_name: str) -> bool:
        """Save a conversation to a file."""
        if not file_name.endswith('.txt'):
            file_name += '.txt'

        file_path = os.path.join(self.chats_directory, file_name)

        try:
            with open(file_path, 'w') as file:
                for message in conversation_history:
                    file.write(f"{message['role']}: {message['content']}\n")
            return True
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")
            return False


class RagChatbot:
    """Main class that orchestrates the RAG chatbot functionality."""

    def __init__(self,
                llm_model: str = 'deepseek-r1:8b',
                embedding_model: str = 'nomic-embed-text',
                persist_directory: str = "./vector_db",
                chats_directory: str = "chats"):
        """Initialize the RAG chatbot."""
        self.llm_model = llm_model
        self.vectorstore = VectorStore(persist_directory, embedding_model)
        self.websearch = WebSearch()
        self.conversation_manager = ConversationManager(chats_directory)
        self.conversation_history = [{"role": "system", "content": "You are a helpful AI assistant."}]

        # Feature toggles
        self.rag_enabled = True
        self.web_search_enabled = True

    def process_command(self, command: str) -> str:
        """Process system commands."""
        command = command.strip().lower()

        if command == "!websearch on":
            self.web_search_enabled = True
            return "Web search enabled"
        elif command == "!websearch off":
            self.web_search_enabled = False
            return "Web search disabled"
        elif command == "!rag on":
            self.rag_enabled = True
            return "RAG retrieval enabled"
        elif command == "!rag off":
            self.rag_enabled = False
            return "RAG retrieval disabled"
        elif command.startswith("!status"):
            return f"Status: RAG: {'ON' if self.rag_enabled else 'OFF'}, Web Search: {'ON' if self.web_search_enabled else 'OFF'}"
        else:
            return ""

    def get_response(self, query: str) -> Iterator[Dict[str, Any]]:
        """Get response for a user query using RAG pipeline."""
        logger.info(f"Processing query: {query[:50]}...")

        # Initialize contexts
        local_context = ""
        web_context = ""

        # Get local context if RAG is enabled
        if self.rag_enabled:
            local_context = self.vectorstore.retrieve_context(query)
            logger.info(f"Retrieved local context: {local_context[:100]}...")

        # Get web context if web search is enabled and needed
        if self.web_search_enabled and self.websearch.should_search_web(query, local_context):
            logger.info("Searching the web...")
            web_context = self.websearch.get_web_context(query)
            logger.info(f"Retrieved web context: {web_context[:100]}...")

        # Generate response
        return self._generate_response(query, local_context, web_context)

    def _generate_response(self, query: str, local_context: str, web_context: str) -> Iterator[Dict[str, Any]]:
        """Generate response using Ollama."""
        try:
            context = self._prepare_context(local_context, web_context)

            formatted_prompt = f"""Question: {query}

Context: {context}

Answer the question based on the provided context. If the information is from web search, 
mention that in your response. If you don't know the answer, say so clearly."""

            response = ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': formatted_prompt}],
                stream=True
            )
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return iter([{'message': {'content': f"Error: {str(e)}"}}])

    def _prepare_context(self, local_context: str, web_context: str) -> str:
        """Prepare combined context from local and web sources."""
        if not local_context and not web_context:
            return "No relevant information found in the knowledge base or web search."
        elif not local_context:
            return f"Web search provided: {web_context}"
        elif not web_context:
            return local_context
        else:
            return f"From local knowledge base: {local_context}\n\nFrom web search: {web_context}"

    def run_interactive(self):
        """Run the chatbot in interactive mode."""
        # Check for saved conversations
        saved_conversations = self.conversation_manager.list_conversations()
        if saved_conversations:
            print("Saved conversations:")
            for i, conv in enumerate(saved_conversations):
                print(f"{i+1}. {conv}")
            option = input("Load saved conversation or start new? (load/new): ")
            if option.lower() == "load":
                try:
                    idx = int(input("Enter the number to load: ")) - 1
                    self.conversation_history = self.conversation_manager.load_conversation(saved_conversations[idx])
                    print(f"Loaded conversation: {saved_conversations[idx]}")
                except (ValueError, IndexError) as e:
                    logger.error(f"Error loading conversation: {str(e)}")
                    print("Invalid selection. Starting new conversation.")
            else:
                print("Starting new conversation.")

        # Set initial configurations
        search_option = input("Enable web search capability? (yes/no, default: yes): ").lower()
        self.web_search_enabled = search_option != "no"
        
        rag_option = input("Enable RAG retrieval? (yes/no, default: yes): ").lower()
        self.rag_enabled = rag_option != "no"
        
        print(f"Features - RAG: {'enabled' if self.rag_enabled else 'disabled'}, "
              f"Web search: {'enabled' if self.web_search_enabled else 'disabled'}")
        print("Commands: !websearch on/off, !rag on/off, !status, quit")
        print("Type ',,,' on a new line to submit multi-line input")
        
        # Main conversation loop
        while True:
            user_input = ""
            print("> ", end="", flush=True)
            
            while True:
                line = input() if not user_input else input("... ")
                
                if line == ",,,":
                    break
                elif line.startswith("!"):
                    command_response = self.process_command(line)
                    if command_response:
                        print(command_response)
                    user_input = ""
                    break
                elif line.lower() == "quit":
                    user_input = "quit"
                    break
                else:
                    user_input += line + "\n"
            
            if not user_input.strip():
                continue
                
            if user_input.lower().strip() == "quit":
                break

            self.conversation_history.append({"role": "user", "content": user_input.strip()})
            print("\nProcessing...", end="\r")
            
            response_iterator = self.get_response(user_input.strip())
            print(" " * 12, end="\r")  # Clear "Processing..." message
            
            assistant_response = ""
            print("Assistant: ", end="", flush=True)
            for chunk in response_iterator:
                chunk_content = chunk['message']['content']
                assistant_response += chunk_content
                print(chunk_content, end="", flush=True)
            print("\n")  # Add a newline after response
            
            self.conversation_history.append({"role": "assistant", "content": assistant_response.strip()})

        # Save conversation on exit
        save_opt = input("Save this conversation? (yes/no): ")
        if save_opt.lower() == "yes":
            file_name = input("Enter conversation file name: ").strip()
            if file_name:
                success = self.conversation_manager.save_conversation(self.conversation_history, file_name)
                if success:
                    print(f"Conversation saved as {file_name}")
                else:
                    print("Failed to save conversation")


if __name__ == "__main__":
    # Create and run the chatbot
    chatbot = RagChatbot()
    chatbot.run_interactive()

