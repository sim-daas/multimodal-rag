# MultiModal RAG Chatbot

A powerful, extensible Retrieval Augmented Generation (RAG) chatbot with multimodal capabilities, built with FastAPI and Ollama.

## Features

### Core Capabilities
- **Retrieval Augmented Generation (RAG)**: Enhance responses with relevant context from your document knowledge base
- **Web Search Integration**: Dynamically retrieve information from the web for up-to-date responses
- **Conversation Memory**: Maintain context across multi-turn conversations
- **Session Management**: Support for multiple concurrent chat sessions
- **Conversation History**: Save and load conversation histories

### Multimodal Support
- **Text Documents**: Process and retrieve information from PDFs, text files, code files, and more
- **Image Understanding**: Process images using LLaVA vision-language model with two modes:
  - **Description Mode**: Extract detailed descriptions from images to enhance text-based context
  - **Direct Image Mode**: Answer questions directly about images using the multimodal LLaVA model
- **Audio Processing**: Transcribe audio files to text for seamless voice interactions

### Advanced Features
- **Streaming Responses**: Real-time streaming of responses for better UX
- **Feature Toggles**: Easily enable/disable RAG, Web Search, and Image Mode
- **Document Analysis**: Extract and analyze content from various file types
- **API-First Design**: Complete REST API for integration with any frontend

## Architecture

The MultiModal RAG Chatbot consists of several key components:

- **RagChatbot**: Core orchestration class that manages the overall chatbot functionality
- **VectorStore**: Manages document embeddings and retrieval using Chroma and Ollama embeddings
- **DocumentProcessor**: Processes various document types and prepares them for embedding
- **WebSearch**: Handles web search functionality and content extraction
- **ImageProcessor**: Processes images using the LLaVA vision-language model
- **AudioProcessor**: Transcribes audio files using Groq API
- **ConversationManager**: Handles saving and loading conversation history

## Setup

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running locally (for LLM and embeddings)
- Optional: Groq API key for audio transcription

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multimodal-rag.git
cd multimodal-rag
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Pull required models with Ollama:
```bash
# Pull the text LLM model
ollama pull deepseek-r1:8b

# Pull the embedding model
ollama pull nomic-embed-text

# Pull the vision-language model for image processing
ollama pull llava:13b
```

5. Set environment variables (optional):
```bash
# For audio processing:
export GROQ_API_KEY=your_groq_api_key
```

### Starting the Server

Run the API server:
```bash
python api.py
```

The server will be available at http://localhost:8000.

## Usage

### API Endpoints

#### Chat Interactions
- `POST /chat`: Send a message and get a response
- `POST /chat/stream`: Stream a chat response in real-time
- `POST /audio/transcribe`: Transcribe audio to text
- `POST /audio/chat`: Process audio into text and get a chat response
- `POST /images/upload`: Upload and process an image

#### Session Management
- `GET /sessions`: List all active sessions
- `GET /sessions/{session_id}`: Get session information
- `POST /sessions/{session_id}/history`: Get conversation history
- `DELETE /sessions/{session_id}`: Delete a session

#### Feature Control
- `POST /features/toggle`: Toggle features (RAG, web search, image mode)

#### Document Management
- `POST /documents/upload`: Upload a document to the vector store
- `POST /documents/process-list`: Process documents from a list file

#### Conversation Management
- `GET /conversations`: List saved conversations
- `POST /conversations/save`: Save the current conversation
- `POST /conversations/load`: Load a saved conversation

### Interactive CLI

You can also use the chatbot in interactive CLI mode:

```bash
python newclass.py
```

## Configuration Options

### RagChatbot Configuration
- `llm_model`: LLM model to use (default: 'deepseek-r1:8b')
- `embedding_model`: Embedding model for vector storage (default: 'nomic-embed-text')
- `persist_directory`: Where to store vector database (default: './vector_db')
- `llava_model`: Vision-language model for image processing (default: 'llava:13b')

### Feature Toggles
Control various features through the API or CLI:
- `rag_enabled`: Enable/disable RAG retrieval
- `web_search_enabled`: Enable/disable web search
- `image_mode_enabled`: Toggle between description mode and direct image mode

## Example: Processing an Image

1. Upload an image:
```bash
curl -X POST "http://localhost:8000/images/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "session_id=your_session_id" \
  -F "file=@/path/to/your/image.jpg"
```

2. Chat with the image context (in description mode):
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"message":"What can you tell me about the image I uploaded?","session_id":"your_session_id"}'
```

3. Toggle to direct image mode:
```bash
curl -X POST "http://localhost:8000/features/toggle" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"your_session_id","feature":"image","enabled":true}'
```

4. Ask a question directly about the image:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"message":"What colors are in this image?","session_id":"your_session_id"}'
```

## License

[MIT License](LICENSE)

## Acknowledgements

This project uses several open-source libraries and models:
- [FastAPI](https://fastapi.tiangolo.com/)
- [Ollama](https://ollama.ai/)
- [LangChain](https://www.langchain.com/)
- [Groq](https://groq.com/)
- [LLaVA](https://llava-vl.github.io/)
