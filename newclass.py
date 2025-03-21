#!/usr/bin/env python

import os
import ollama
import re
import trafilatura
import requests
import json
import time
import logging
import csv
import pathlib
from typing import List, Dict, Any, Optional, Iterator, Tuple, Set, Union
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader
)
import tqdm
from groq import Groq

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ragchatbot")

class DocumentProcessor:
    """Processes various document types and prepares them for embedding."""
    
    # Supported file extensions by category
    SUPPORTED_EXTENSIONS = {
        "pdf": [".pdf"],
        "text": [".txt", ".md", ".log"],
        "code": [".py", ".js", ".java", ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".html", ".css"],
        "data": [".csv", ".json", ".xml"],
        "web": [".html", ".htm"]
    }
    
    def __init__(self, 
                 vector_store: 'VectorStore',
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200):
        """Initialize the document processor with a vector store."""
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        self.successful_docs = 0
        self.failed_docs = 0
    
    def read_document_paths(self, file_list_path: str) -> List[str]:
        """Read a list of document paths from a file."""
        document_paths = []
        try:
            with open(file_list_path, 'r') as file:
                for line in file:
                    path = line.strip()
                    if path and not path.startswith('#'):  # Skip empty lines and comments
                        document_paths.append(path)
            logger.info(f"Read {len(document_paths)} document paths from {file_list_path}")
            return document_paths
        except Exception as e:
            logger.error(f"Error reading document paths from {file_list_path}: {str(e)}")
            return []
    
    def detect_file_type(self, file_path: str) -> str:
        """Detect the type of file based on extension."""
        file_ext = pathlib.Path(file_path).suffix.lower()
        
        for category, extensions in self.SUPPORTED_EXTENSIONS.items():
            if file_ext in extensions:
                return category
        
        return "unknown"
    
    def process_document(self, file_path: str) -> bool:
        """Process a single document and add it to the vector store."""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                self.failed_docs += 1
                return False
            
            file_type = self.detect_file_type(file_path)
            logger.info(f"Processing {file_type} document: {file_path}")
            
            # Extract text content based on file type
            if file_type == "pdf":
                content = self._extract_from_pdf(file_path)
            elif file_type == "text":
                content = self._extract_from_text(file_path)
            elif file_type == "code":
                content = self._extract_from_code(file_path)
            elif file_type == "data":
                content = self._extract_from_data(file_path)
            elif file_type == "web":
                content = self._extract_from_web(file_path)
            else:
                logger.warning(f"Unsupported file type for {file_path}")
                self.failed_docs += 1
                return False
            
            if not content:
                logger.warning(f"No content extracted from {file_path}")
                self.failed_docs += 1
                return False
            
            # Add document to vector store
            self.vector_store.add_document(content, source=file_path)
            self.successful_docs += 1
            return True
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            self.failed_docs += 1
            return False
    
    def process_documents_batch(self, document_paths: List[str]) -> Dict[str, int]:
        """Process a batch of documents with progress reporting."""
        self.successful_docs = 0
        self.failed_docs = 0
        
        logger.info(f"Starting batch processing of {len(document_paths)} documents")
        
        for doc_path in tqdm.tqdm(document_paths, desc="Processing documents"):
            self.process_document(doc_path)
        
        results = {
            "total": len(document_paths),
            "successful": self.successful_docs,
            "failed": self.failed_docs
        }
        
        logger.info(f"Batch processing complete. Successfully processed {self.successful_docs} documents, {self.failed_docs} failed.")
        return results
    
    def process_documents_from_file(self, file_list_path: str) -> Dict[str, int]:
        """Process documents listed in a file."""
        document_paths = self.read_document_paths(file_list_path)
        if not document_paths:
            logger.warning(f"No document paths found in {file_list_path}")
            return {"total": 0, "successful": 0, "failed": 0}
        
        return self.process_documents_batch(document_paths)
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Extract and join text from all pages
            text_content = "\n\n".join([doc.page_content for doc in documents])
            
            # Handle potential OCR issues - remove excessive whitespace
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            return text_content
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            return ""
    
    def _extract_from_text(self, file_path: str) -> str:
        """Extract text from a plain text file."""
        file_ext = pathlib.Path(file_path).suffix.lower()
        
        try:
            if file_ext == ".md":
                loader = UnstructuredMarkdownLoader(file_path)
                documents = loader.load()
                return "\n\n".join([doc.page_content for doc in documents])
            else:  # .txt or other text files
                loader = TextLoader(file_path)
                documents = loader.load()
                return "\n".join([doc.page_content for doc in documents])
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            # Fallback method
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    return file.read()
            except Exception as fallback_e:
                logger.error(f"Fallback extraction failed for {file_path}: {str(fallback_e)}")
                return ""
    
    def _extract_from_code(self, file_path: str) -> str:
        """Extract content from code files, preserving structure and comments."""
        try:
            # Use TextLoader for code files
            loader = TextLoader(file_path)
            documents = loader.load()
            code_content = "\n".join([doc.page_content for doc in documents])
            
            # Add file metadata
            file_name = os.path.basename(file_path)
            file_ext = pathlib.Path(file_path).suffix.lower()
            
            # Enhance the content with metadata
            enhanced_content = f"File: {file_name}\nType: {file_ext[1:]} code\n\n{code_content}"
            
            return enhanced_content
        except Exception as e:
            logger.error(f"Error extracting code from {file_path}: {str(e)}")
            # Fallback method
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    code = file.read()
                return code
            except Exception as fallback_e:
                logger.error(f"Fallback extraction failed for {file_path}: {str(fallback_e)}")
                return ""
    
    def _extract_from_data(self, file_path: str) -> str:
        """Extract content from data files like CSV, JSON."""
        file_ext = pathlib.Path(file_path).suffix.lower()
        
        try:
            if file_ext == ".csv":
                # For CSV, we want to create a readable representation
                result = []
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    csv_reader = csv.reader(file)
                    headers = next(csv_reader, None)
                    if headers:
                        result.append("Headers: " + ", ".join(headers))
                        for i, row in enumerate(csv_reader):
                            if i < 10:  # Only include first 10 rows for context
                                result.append(" | ".join(row))
                            else:
                                result.append("... (additional rows omitted)")
                                break
                return "\n".join(result)
            
            elif file_ext == ".json":
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    data = json.load(file)
                    # Return a formatted string representation
                    return json.dumps(data, indent=2)
            
            elif file_ext == ".xml":
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    return file.read()
            
            else:
                # For other data files, read as text
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    return file.read()
        
        except Exception as e:
            logger.error(f"Error extracting data from {file_path}: {str(e)}")
            return ""
    
    def _extract_from_web(self, file_path: str) -> str:
        """Extract content from HTML files."""
        try:
            loader = UnstructuredHTMLLoader(file_path)
            documents = loader.load()
            content = "\n\n".join([doc.page_content for doc in documents])
            
            # Clean up HTML content
            content = re.sub(r'\s+', ' ', content).strip()
            
            return content
        except Exception as e:
            logger.error(f"Error extracting content from HTML {file_path}: {str(e)}")
            
            # Fallback using BeautifulSoup
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    html_content = file.read()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.extract()
                    
                    # Get text
                    text = soup.get_text()
                    
                    # Break into lines and remove leading/trailing space
                    lines = (line.strip() for line in text.splitlines())
                    # Break multi-headlines into a line each
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    # Remove blank lines
                    text = '\n'.join(chunk for chunk in chunks if chunk)
                    
                    return text
            except Exception as fallback_e:
                logger.error(f"Fallback HTML extraction failed for {file_path}: {str(fallback_e)}")
                return ""


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
    
    def should_search_web(self, query: str, local_context: str) -> bool:
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


class AudioProcessor:
    """Processes audio files and transcribes them using Groq API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the audio processor with optional API key."""
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("No Groq API key provided. Audio transcription will not work.")
        
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.supported_formats = [".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"]
    
    def is_supported_format(self, filename: str) -> bool:
        """Check if the file format is supported for transcription."""
        file_ext = pathlib.Path(filename).suffix.lower()
        return file_ext in self.supported_formats
    
    def transcribe(self, file_path: str) -> Dict[str, Any]:
        """Transcribe an audio file using Groq API."""
        if not self.client:
            logger.error("Cannot transcribe audio: Groq client not initialized")
            return {"success": False, "error": "Groq API key not configured"}
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"success": False, "error": "File not found"}
            
        if not self.is_supported_format(file_path):
            logger.error(f"Unsupported file format: {file_path}")
            return {"success": False, "error": "Unsupported file format"}
        
        try:
            with open(file_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(os.path.basename(file_path), file.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                )
            
            return {
                "success": True,
                "text": transcription.text,
                "metadata": {
                    "duration": getattr(transcription, "duration", None),
                    "language": getattr(transcription, "language", None)
                }
            }
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return {"success": False, "error": str(e)}


class RagChatbot:
    """Main class that orchestrates the RAG chatbot functionality."""

    def __init__(self,
                llm_model: str = 'deepseek-r1:8b',
                embedding_model: str = 'nomic-embed-text',
                persist_directory: str = "./vector_db",
                chats_directory: str = "chats",
                chunk_size: int = 1000,
                chunk_overlap: int = 200,
                groq_api_key: Optional[str] = None):
        """Initialize the RAG chatbot."""
        self.llm_model = llm_model
        self.vectorstore = VectorStore(persist_directory, embedding_model)
        self.websearch = WebSearch()
        self.conversation_manager = ConversationManager(chats_directory)
        self.document_processor = DocumentProcessor(
            vector_store=self.vectorstore,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.conversation_history = [{"role": "system", "content": "You are a helpful AI assistant."}]
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(api_key=groq_api_key)

        # Feature toggles
        self.rag_enabled = True
        self.web_search_enabled = True
        self.audio_enabled = True
    
    def process_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """Process an audio file and return its transcription."""
        if not self.audio_enabled:
            return {"success": False, "error": "Audio processing is disabled"}
        
        result = self.audio_processor.transcribe(audio_file_path)
        return result
    
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
        elif command.startswith("!process "):
            # Process documents from a file list
            file_path = command[9:].strip()
            if os.path.exists(file_path):
                print(f"Processing documents from {file_path}...")
                results = self.document_processor.process_documents_from_file(file_path)
                return f"Processed {results['successful']} documents successfully, {results['failed']} failed"
            else:
                return f"File not found: {file_path}"
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
        if self.web_search_enabled:
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
        print("Commands: !websearch on/off, !rag on/off, !status, !process <file_list.txt>, quit")
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

