#!/usr/bin/env python

import os
import uuid
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio


import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, File, UploadFile, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, EmailStr
from groq import Groq

from newclass import RagChatbot, VectorStore, WebSearch, ConversationManager, AudioProcessor

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="API for interacting with a Retrieval Augmented Generation (RAG) chatbot",
    version="1.0.0"
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure a logger for API requests
api_logger = logging.getLogger("api")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
api_logger.addHandler(handler)
api_logger.setLevel(logging.INFO)

# Session management
sessions: Dict[str, RagChatbot] = {}
session_last_active: Dict[str, datetime] = {}
SESSION_TIMEOUT_MINUTES = 30

# ----- Pydantic Models -----

class Message(BaseModel):
    """A chat message model."""
    role: str = Field(..., description="The role of the message sender (user, assistant, or system)")
    content: str = Field(..., description="The content of the message")

class ChatRequest(BaseModel):
    """Model for chat message request."""
    message: str = Field(..., description="User message to send to the chatbot")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")

class ChatResponse(BaseModel):
    """Model for chat message response."""
    message: str = Field(..., description="Assistant's response message")
    session_id: str = Field(..., description="Session ID for conversation continuity")

class SessionInfo(BaseModel):
    """Information about a chatbot session."""
    session_id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(..., description="When the session was created")
    last_active: datetime = Field(..., description="When the session was last used")
    rag_enabled: bool = Field(..., description="Whether RAG retrieval is enabled")
    web_search_enabled: bool = Field(..., description="Whether web search is enabled")
    image_mode_enabled: bool = Field(False, description="Whether image mode is enabled")
    message_count: int = Field(..., description="Number of messages in the conversation")

class FeatureToggleRequest(BaseModel):
    """Request model for toggling features."""
    session_id: str = Field(..., description="Session ID for the conversation")
    feature: str = Field(..., description="Feature to toggle: 'rag', 'websearch', or 'image'")
    enabled: bool = Field(..., description="Whether to enable or disable the feature")

class DocumentUploadResponse(BaseModel):
    """Response after document upload."""
    success: bool = Field(..., description="Whether the document was successfully processed")
    filename: str = Field(..., description="Name of the processed file")
    message: str = Field(..., description="Processing status message")

class ConversationListResponse(BaseModel):
    """Response for listing saved conversations."""
    conversations: List[str] = Field(..., description="List of saved conversation names")

class ConversationSaveRequest(BaseModel):
    """Request to save a conversation."""
    session_id: str = Field(..., description="Session ID of the conversation to save")
    filename: str = Field(..., description="Filename to save the conversation as")

class ConversationLoadRequest(BaseModel):
    """Request to load a conversation."""
    conversation_name: str = Field(..., description="Name of the conversation to load")

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed explanation")

class AudioTranscriptionResponse(BaseModel):
    """Response after audio transcription."""
    success: bool = Field(..., description="Whether the audio was successfully transcribed")
    text: Optional[str] = Field(None, description="Transcribed text")
    session_id: str = Field(..., description="Session ID for conversation continuity")
    error: Optional[str] = Field(None, description="Error message if transcription failed")
    duration: Optional[float] = Field(None, description="Duration of the audio in seconds")
    language: Optional[str] = Field(None, description="Detected language of the audio")

class ImageUploadResponse(BaseModel):
    """Response after image upload."""
    success: bool = Field(..., description="Whether the image was successfully processed")
    image_id: Optional[str] = Field(None, description="Unique ID for the uploaded image")
    filename: str = Field(..., description="Name of the processed file")
    description: Optional[str] = Field(None, description="AI-generated description of the image")
    message: str = Field(..., description="Processing status message")
    error: Optional[str] = Field(None, description="Error message if processing failed")

# ----- Helper Functions -----

def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, RagChatbot]:
    """Get an existing session or create a new one."""
    if session_id and session_id in sessions:
        # Update last active timestamp
        session_last_active[session_id] = datetime.now()
        return session_id, sessions[session_id]
    
    # Create new session
    new_session_id = str(uuid.uuid4())
    sessions[new_session_id] = RagChatbot()
    session_last_active[new_session_id] = datetime.now()
    return new_session_id, sessions[new_session_id]

def cleanup_inactive_sessions():
    """Remove sessions that have been inactive for too long."""
    cutoff_time = datetime.now() - timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    inactive_sessions = [
        session_id for session_id, last_active in session_last_active.items() 
        if last_active < cutoff_time
    ]
    
    for session_id in inactive_sessions:
        if session_id in sessions:
            del sessions[session_id]
        if session_id in session_last_active:
            del session_last_active[session_id]

# ----- API Endpoints -----

@app.get("/", response_model=dict)
async def root():
    """Root endpoint to check API status."""
    return {
        "status": "online",
        "message": "RAG Chatbot API is running",
        "version": "1.0.0"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Send a message to the chatbot and get a response."""
    # Clean up old sessions periodically
    background_tasks.add_task(cleanup_inactive_sessions)
    
    # Get or create a session
    session_id, chatbot = get_or_create_session(request.session_id)
    
    try:
        # Add user message to conversation history
        chatbot.conversation_history.append({"role": "user", "content": request.message})
        
        # Process message with RAG pipeline
        response_iterator = chatbot.get_response(request.message)
        
        # Collect response
        assistant_response = ""
        for chunk in response_iterator:
            assistant_response += chunk['message']['content']
        
        # Add assistant response to conversation history
        chatbot.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        return ChatResponse(
            message=assistant_response,
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )

@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """Stream a chat response for realtime UI updates."""
    api_logger.info(f"Received streaming chat request: {request.message[:50]}...")
    
    session_id, chatbot = get_or_create_session(request.session_id)
    api_logger.info(f"Using session: {session_id} for streaming request")
    
    try:
        # Add user message to conversation history
        chatbot.conversation_history.append({"role": "user", "content": request.message})
        api_logger.info(f"Added user message to history, now generating response")
        
        # Get streaming response iterator from the chatbot
        response_iterator = chatbot.get_response(request.message)
        
        # Create an async generator for streaming the response
        async def generate():
            full_response = ""
            chunk_count = 0
            
            try:
                # Process each chunk from the iterator
                for chunk in response_iterator:
                    chunk_count += 1
                    content = chunk['message']['content']
                    full_response += content
                    api_logger.info(f"Sending chunk {chunk_count}: {content[:20]}...")
                    
                    # Format as server-sent event
                    yield f"data: {json.dumps({'content': content, 'full_response': full_response, 'session_id': session_id})}\n\n"
                    await asyncio.sleep(0.01)  # Small delay to ensure proper streaming
                
                # After completion, add response to history
                chatbot.conversation_history.append({"role": "assistant", "content": full_response})
                api_logger.info(f"Completed streaming response with {chunk_count} chunks. Total length: {len(full_response)}")
            except Exception as e:
                api_logger.error(f"Error during response generation: {str(e)}")
                # Send error message to client
                yield f"data: {json.dumps({'error': str(e), 'session_id': session_id})}\n\n"
        
        # Return streaming response
        return StreamingResponse(
            generate(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable buffering for Nginx
            }
        )
    except Exception as e:
        api_logger.error(f"Error in chat stream endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )

@app.get("/api/healthcheck", response_model=dict)
async def healthcheck():
    """Health check endpoint for the frontend to test connectivity."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/sessions", response_model=List[SessionInfo])
async def list_sessions():
    """List all active sessions."""
    result = []
    for session_id, chatbot in sessions.items():
        result.append(SessionInfo(
            session_id=session_id,
            created_at=session_last_active[session_id] - timedelta(minutes=5),  # Estimate
            last_active=session_last_active[session_id],
            rag_enabled=chatbot.rag_enabled,
            web_search_enabled=chatbot.web_search_enabled,
            image_mode_enabled=chatbot.image_mode_enabled,
            message_count=len([m for m in chatbot.conversation_history if m["role"] != "system"])
        ))
    return result

@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get information about a specific session."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    chatbot = sessions[session_id]
    return SessionInfo(
        session_id=session_id,
        created_at=session_last_active[session_id] - timedelta(minutes=5),  # Estimate
        last_active=session_last_active[session_id],
        rag_enabled=chatbot.rag_enabled,
        web_search_enabled=chatbot.web_search_enabled,
        image_mode_enabled=chatbot.image_mode_enabled,
        message_count=len([m for m in chatbot.conversation_history if m["role"] != "system"])
    )

@app.post("/sessions/{session_id}/history", response_model=List[Message])
async def get_conversation_history(session_id: str):
    """Get the conversation history for a session."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    chatbot = sessions[session_id]
    return [
        Message(role=msg["role"], content=msg["content"])
        for msg in chatbot.conversation_history
    ]

@app.post("/features/toggle", response_model=dict)
async def toggle_feature(request: FeatureToggleRequest):
    """Toggle RAG, web search, or image mode features."""
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {request.session_id} not found"
        )
    
    chatbot = sessions[request.session_id]
    
    if request.feature.lower() == "rag":
        chatbot.rag_enabled = request.enabled
        return {
            "success": True,
            "feature": "rag",
            "enabled": chatbot.rag_enabled,
            "message": f"RAG retrieval {'enabled' if request.enabled else 'disabled'}"
        }
    elif request.feature.lower() == "websearch":
        chatbot.web_search_enabled = request.enabled
        return {
            "success": True,
            "feature": "websearch",
            "enabled": chatbot.web_search_enabled,
            "message": f"Web search {'enabled' if request.enabled else 'disabled'}"
        }
    elif request.feature.lower() == "image":
        chatbot.image_mode_enabled = request.enabled
        return {
            "success": True,
            "feature": "image",
            "enabled": chatbot.image_mode_enabled,
            "message": f"Image mode {'enabled' if request.enabled else 'disabled'}"
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown feature: {request.feature}. Supported features: 'rag', 'websearch', 'image'"
        )

@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload a document to add to the vector store."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    chatbot = sessions[session_id]
    
    # Create a temporary file to save the upload
    temp_file_path = f"./temp_{file.filename}"
    
    try:
        # Save the uploaded file
        contents = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(contents)
        
        # Process document using DocumentProcessor
        document_processor = chatbot.document_processor
        
        # Detect file type
        file_type = document_processor.detect_file_type(temp_file_path)
        
        # Extract content based on file type
        content = ""
        if file_type == "pdf":
            content = document_processor._extract_from_pdf(temp_file_path)
        elif file_type == "text":
            content = document_processor._extract_from_text(temp_file_path)
        elif file_type == "code":
            content = document_processor._extract_from_code(temp_file_path)
        elif file_type == "data":
            content = document_processor._extract_from_data(temp_file_path)
        elif file_type == "web":
            content = document_processor._extract_from_web(temp_file_path)
        else:
            # Fallback for unknown file types
            with open(temp_file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        
        if not content:
            return DocumentUploadResponse(
                success=False,
                filename=file.filename,
                message=f"No content could be extracted from {file.filename}"
            )
        
        # Add document to vector store
        chatbot.vectorstore.add_document(content, source=file.filename)
        
        # Schedule cleanup in background
        if background_tasks:
            background_tasks.add_task(lambda: os.remove(temp_file_path) if os.path.exists(temp_file_path) else None)
        
        return DocumentUploadResponse(
            success=True,
            filename=file.filename,
            message=f"Document successfully added to vector store"
        )
    except Exception as e:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@app.post("/documents/process-list", response_model=dict)
async def process_document_list(
    session_id: str,
    document_list_path: str
):
    """Process documents from a list file."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    if not os.path.exists(document_list_path):
        raise HTTPException(
            status_code=400,
            detail=f"Document list file not found: {document_list_path}"
        )
    
    chatbot = sessions[session_id]
    document_processor = None
    
    # Check if DocumentProcessor class is available
    if hasattr(chatbot, 'document_processor'):
        document_processor = chatbot.document_processor
    else:
        # Import here to avoid circular imports
        try:
            from newclass import DocumentProcessor
            document_processor = DocumentProcessor(vector_store=chatbot.vectorstore)
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="DocumentProcessor class not available"
            )
    
    # Process documents
    try:
        results = document_processor.process_documents_from_file(document_list_path)
        return {
            "success": True,
            "processed": results["successful"],
            "failed": results["failed"],
            "total": results["total"],
            "message": f"Processed {results['successful']} documents successfully, {results['failed']} failed"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing documents: {str(e)}"
        )

@app.get("/conversations", response_model=ConversationListResponse)
async def list_conversations():
    """List all saved conversations."""
    conversation_manager = ConversationManager()
    conversations = conversation_manager.list_conversations()
    return ConversationListResponse(
        conversations=conversations
    )

@app.post("/conversations/save", response_model=dict)
async def save_conversation(request: ConversationSaveRequest):
    """Save the current conversation."""
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {request.session_id} not found"
        )
    
    chatbot = sessions[request.session_id]
    conversation_manager = chatbot.conversation_manager
    
    # Save the conversation
    success = conversation_manager.save_conversation(
        chatbot.conversation_history,
        request.filename
    )
    
    if success:
        return {
            "success": True,
            "filename": request.filename,
            "message": f"Conversation saved as {request.filename}"
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to save conversation"
        )

@app.post("/conversations/load", response_model=dict)
async def load_conversation(request: ConversationLoadRequest):
    """Load a saved conversation into a new session."""
    # Create a new session
    session_id, chatbot = get_or_create_session()
    
    try:
        # Load the conversation
        conversation_history = chatbot.conversation_manager.load_conversation(request.conversation_name)
        chatbot.conversation_history = conversation_history
        
        return {
            "success": True,
            "session_id": session_id,
            "conversation_name": request.conversation_name,
            "message": f"Conversation '{request.conversation_name}' loaded into new session"
        }
    except Exception as e:
        # Clean up the session if loading failed
        if session_id in sessions:
            del sessions[session_id]
        if session_id in session_last_active:
            del session_last_active[session_id]
        
        raise HTTPException(
            status_code=500,
            detail=f"Error loading conversation: {str(e)}"
        )

@app.delete("/sessions/{session_id}", response_model=dict)
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    del sessions[session_id]
    if session_id in session_last_active:
        del session_last_active[session_id]
    
    return {
        "success": True,
        "session_id": session_id,
        "message": f"Session {session_id} deleted"
    }

@app.post("/audio/transcribe", response_model=AudioTranscriptionResponse)
async def transcribe_audio(
    session_id: Optional[str] = Form(None),
    file: UploadFile = File(...),
    add_to_conversation: bool = Form(False),
    background_tasks: BackgroundTasks = None
):
    """Transcribe an audio file and optionally add it to the conversation."""
    # Get or create a session
    session_id, chatbot = get_or_create_session(session_id)
    
    # Validate audio file format
    if not chatbot.audio_processor.is_supported_format(file.filename):
        return AudioTranscriptionResponse(
            success=False,
            session_id=session_id,
            error=f"Unsupported audio format. Supported formats: {', '.join(chatbot.audio_processor.supported_formats)}"
        )
    
    # Create a temporary file to save the upload
    temp_file_path = f"./temp_audio_{file.filename}"
    
    try:
        # Save the uploaded file
        contents = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(contents)
        
        # Process audio file
        result = chatbot.process_audio(temp_file_path)
        
        if not result["success"]:
            return AudioTranscriptionResponse(
                success=False,
                session_id=session_id,
                error=result.get("error", "Unknown error during transcription")
            )
        
        transcribed_text = result["text"]
        
        # Add transcription to conversation if requested
        if add_to_conversation and transcribed_text:
            chatbot.conversation_history.append({"role": "user", "content": transcribed_text})
        
        # Schedule cleanup in background
        if background_tasks:
            background_tasks.add_task(lambda: os.remove(temp_file_path) if os.path.exists(temp_file_path) else None)
        
        return AudioTranscriptionResponse(
            success=True,
            text=transcribed_text,
            session_id=session_id,
            duration=result.get("metadata", {}).get("duration"),
            language=result.get("metadata", {}).get("language")
        )
    
    except Exception as e:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return AudioTranscriptionResponse(
            success=False,
            session_id=session_id,
            error=f"Error processing audio: {str(e)}"
        )

@app.post("/audio/chat", response_model=ChatResponse)
async def audio_chat(
    session_id: Optional[str] = Form(None),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Transcribe an audio file, add it to the conversation, and get a response."""
    # Get or create a session
    session_id, chatbot = get_or_create_session(session_id)
    
    # Create a temporary file to save the upload
    temp_file_path = f"./temp_audio_{file.filename}"
    
    try:
        # Save the uploaded file
        contents = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(contents)
        
        # Process audio file
        result = chatbot.process_audio(temp_file_path)
        
        if not result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Audio transcription failed: {result.get('error', 'Unknown error')}"
            )
        
        transcribed_text = result["text"]
        
        # Add transcription to conversation
        chatbot.conversation_history.append({"role": "user", "content": transcribed_text})
        
        # Process message with RAG pipeline
        response_iterator = chatbot.get_response(transcribed_text)
        
        # Collect response
        assistant_response = ""
        for chunk in response_iterator:
            assistant_response += chunk['message']['content']
        
        # Add assistant response to conversation history
        chatbot.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        # Schedule cleanup in background
        if background_tasks:
            background_tasks.add_task(lambda: os.remove(temp_file_path) if os.path.exists(temp_file_path) else None)
        
        return ChatResponse(
            message=assistant_response,
            session_id=session_id
        )
    
    except HTTPException:
        # Clean up temp file and re-raise the exception
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise
    
    except Exception as e:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio chat: {str(e)}"
        )

@app.post("/images/upload", response_model=ImageUploadResponse)
async def upload_image(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload an image to process with the image model."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    chatbot = sessions[session_id]
    
    # Check if file is an image
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        return ImageUploadResponse(
            success=False,
            filename=file.filename,
            message="Uploaded file is not a recognized image format",
            error="Invalid file type"
        )
    
    # Create a temporary file to save the upload
    image_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    temp_file_path = f"./temp_image_{image_id}{file_extension}"
    
    try:
        # Save the uploaded file
        contents = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(contents)
        
        # Process the image using the ImageProcessor
        result = chatbot.process_image(temp_file_path, image_id)
        
        if not result["success"]:
            return ImageUploadResponse(
                success=False,
                filename=file.filename,
                message=f"Failed to process image: {result.get('error', 'Unknown error')}",
                error=result.get("error")
            )
        
        # Schedule cleanup in background (but keep image for the session)
        # We don't want to delete the image as it's needed for future queries in image mode
        
        return ImageUploadResponse(
            success=True,
            image_id=image_id,
            filename=file.filename,
            description=result.get("description"),
            message="Image successfully processed and added to conversation context"
        )
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return ImageUploadResponse(
            success=False,
            filename=file.filename,
            message=f"Error processing image: {str(e)}",
            error=str(e)
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Run the application
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
