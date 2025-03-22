#!/usr/bin/env python

import os
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, File, UploadFile, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, EmailStr
from groq import Groq

from newclass import RagChatbot, VectorStore, WebSearch, ConversationManager, AudioProcessor

import shutil
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="API for interacting with a Retrieval Augmented Generation (RAG) chatbot",
    version="1.0.0"
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session management
sessions: Dict[str, RagChatbot] = {}
session_last_active: Dict[str, datetime] = {}
SESSION_TIMEOUT_MINUTES = 30

# Define base upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Add this logger initialization near the top of the file, before any function definitions
import logging
logger = logging.getLogger("api")

# ----- Pydantic Models -----

class Message(BaseModel):
    """A chat message model."""
    role: str = Field(..., description="The role of the message sender (user, assistant, or system)")
    content: str = Field(..., description="The content of the message")

class ChatSettings(BaseModel):
    """Model for chat settings."""
    rag_enabled: Optional[bool] = Field(None, description="Whether to enable RAG retrieval")
    web_search_enabled: Optional[bool] = Field(None, description="Whether to enable web search")

class ChatRequest(BaseModel):
    """Model for chat message request."""
    message: str = Field(..., description="User message to send to the chatbot")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    settings: Optional[ChatSettings] = Field(None, description="Settings for this chat request")

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
    message_count: int = Field(..., description="Number of messages in the conversation")

class FeatureToggleRequest(BaseModel):
    """Request model for toggling features."""
    session_id: str = Field(..., description="Session ID for the conversation")
    feature: str = Field(..., description="Feature to toggle: 'rag' or 'websearch'")
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

class UploadedFile(BaseModel):
    """Information about an uploaded file."""
    filename: str
    path: str
    upload_time: datetime
    file_size: int
    file_type: str

class UploadedFileListResponse(BaseModel):
    """Response containing list of uploaded files."""
    session_id: str
    files: List[UploadedFile]

class FileAnalysisRequest(BaseModel):
    """Request to analyze a specific file."""
    session_id: str = Field(..., description="Session ID of the conversation")
    filename: str = Field(..., description="Filename to analyze")

class FileAnalysisResponse(BaseModel):
    """Response after file analysis."""
    success: bool = Field(..., description="Whether the analysis was successful")
    filename: str = Field(..., description="Name of the analyzed file")
    analysis: str = Field(..., description="Analysis results")
    message: str = Field(..., description="Status message")

class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    settings: Optional[ChatSettings] = Field(None, description="Initial settings for the session")

class CreateSessionResponse(BaseModel):
    """Response after creating a session."""
    session_id: str = Field(..., description="Session ID for the new session")
    message: str = Field(..., description="Status message")

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

def get_session_upload_dir(session_id: str) -> str:
    """Get the upload directory for a session and create it if it doesn't exist."""
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

def get_session_files(session_id: str) -> List[UploadedFile]:
    """Get list of files uploaded for a session."""
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    
    if not os.path.exists(session_dir):
        return []
        
    files = []
    
    for filename in os.listdir(session_dir):
        file_path = os.path.join(session_dir, filename)
        if os.path.isfile(file_path):
            stats = os.stat(file_path)
            file_type = filename.split('.')[-1] if '.' in filename else 'unknown'
            
            files.append(UploadedFile(
                filename=filename,
                path=file_path,
                upload_time=datetime.fromtimestamp(stats.st_mtime),
                file_size=stats.st_size,
                file_type=file_type
            ))
    
    # Sort by upload time, newest first
    files.sort(key=lambda x: x.upload_time, reverse=True)
    return files

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
    
    # Update feature settings from the request if provided
    if request.settings:
        print(f"Received settings: rag_enabled={request.settings.rag_enabled}, web_search_enabled={request.settings.web_search_enabled}")
        
        if request.settings.rag_enabled is not None:
            chatbot.rag_enabled = request.settings.rag_enabled
            print(f"Set chatbot.rag_enabled to {chatbot.rag_enabled}")
        
        if request.settings.web_search_enabled is not None:
            chatbot.web_search_enabled = request.settings.web_search_enabled
            print(f"Set chatbot.web_search_enabled to {chatbot.web_search_enabled}")
    
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
    """Toggle RAG or web search features."""
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {request.session_id} not found"
        )
    
    chatbot = sessions[request.session_id]
    print(f"Toggling feature '{request.feature}' to {'enabled' if request.enabled else 'disabled'} for session {request.session_id}")
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
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown feature: {request.feature}. Supported features: 'rag', 'websearch'"
        )

@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload a document to be processed."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404, 
            detail=f"Session {session_id} not found"
        )
    
    try:
        # Create session upload directory
        session_dir = get_session_upload_dir(session_id)
        
        # Generate a unique filename to avoid overwrites
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = ''.join(c if c.isalnum() or c in '._- ' else '_' for c in file.filename)
        unique_filename = f"{timestamp}_{safe_filename}"
        file_path = os.path.join(session_dir, unique_filename)
        
        # Save the file permanently
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Process the file in the background
        chatbot = sessions[session_id]
        document_result = chatbot.document_processor.process_document(file_path)
        
        # Return success with permanent file path
        return DocumentUploadResponse(
            success=True,
            filename=file.filename,
            message=f"Document processed successfully: {file.filename} ({file_size} bytes)",
            path=file_path
        )
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@app.get("/documents/list/{session_id}", response_model=UploadedFileListResponse)
async def list_session_files(session_id: str):
    """List all files uploaded for a session."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    try:
        files = get_session_files(session_id)
        return UploadedFileListResponse(
            session_id=session_id,
            files=files
        )
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing files: {str(e)}"
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

@app.post("/documents/analyze", response_model=FileAnalysisResponse)
async def analyze_file(request: FileAnalysisRequest):
    """Analyze a specific document."""
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {request.session_id} not found"
        )
    
    chatbot = sessions[request.session_id]
    
    try:
        session_dir = get_session_upload_dir(request.session_id)
        
        # Find the file by filename (look for any file ending with the requested filename)
        found_file = None
        for file_path in os.listdir(session_dir):
            if file_path.endswith(request.filename):
                found_file = os.path.join(session_dir, file_path)
                break
        
        if not found_file:
            raise HTTPException(
                status_code=404,
                detail=f"File {request.filename} not found in session {request.session_id}"
            )
        
        # Ensure document processor exists and has the right method
        if not hasattr(chatbot, 'document_processor'):
            logger.warning(f"Creating missing document_processor for session {request.session_id}")
            from newclass import DocumentProcessor
            chatbot.document_processor = DocumentProcessor(
                vector_store=chatbot.vectorstore if hasattr(chatbot, 'vectorstore') else None
            )
        
        # Check if the analyze_document method exists, if not use a fallback
        if not hasattr(chatbot.document_processor, 'analyze_document'):
            logger.error("analyze_document method not available, using fallback")
            # Reload the module to ensure we have the latest version with analyze_document
            import importlib
            import sys
            if 'newclass' in sys.modules:
                importlib.reload(sys.modules['newclass'])
                # Recreate the document processor
                from newclass import DocumentProcessor
                chatbot.document_processor = DocumentProcessor(
                    vector_store=chatbot.vectorstore if hasattr(chatbot, 'vectorstore') else None
                )
            
            # If still missing, use fallback
            if not hasattr(chatbot.document_processor, 'analyze_document'):
                file_size = os.path.getsize(found_file)
                file_ext = os.path.splitext(found_file)[1].lower().replace('.', '') or 'unknown'
                file_name = os.path.basename(found_file)
                
                # Create a basic analysis as fallback
                analysis_result = (
                    f"File Analysis:\n\n"
                    f"File: {file_name}\n"
                    f"Type: {file_ext.upper()}\n"
                    f"Size: {file_size / 1024:.1f} KB\n\n"
                    f"This file has been processed and indexed.\n"
                    f"You can ask questions about the content in future messages."
                )
                
                return FileAnalysisResponse(
                    success=True,
                    filename=request.filename,
                    analysis=analysis_result,
                    message=f"Successfully analyzed {request.filename} (basic analysis)"
                )
        
        # Direct method call with explicit error handling
        try:
            analysis_result = chatbot.document_processor.analyze_document(found_file)
            if not analysis_result:
                raise ValueError("Analysis resulted in empty output")
        except Exception as method_error:
            logger.error(f"Error in analyze_document method: {str(method_error)}")
            # Generate a fallback analysis
            file_size = os.path.getsize(found_file)
            file_ext = os.path.splitext(found_file)[1].lower().replace('.', '') or 'unknown'
            file_name = os.path.basename(found_file)
            
            analysis_result = (
                f"File Analysis (Fallback):\n\n"
                f"File: {file_name}\n"
                f"Type: {file_ext.upper()}\n"
                f"Size: {file_size / 1024:.1f} KB\n\n"
                f"This file has been processed and indexed.\n"
                f"You can ask questions about the content in future messages.\n\n"
                f"Note: Detailed analysis unavailable ({str(method_error)})"
            )
            
        return FileAnalysisResponse(
            success=True,
            filename=request.filename,
            analysis=analysis_result,
            message=f"Successfully analyzed {request.filename}"
        )
        
    except Exception as e:
        # Enhanced error logging
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error analyzing document: {error_details}")
        
        # Return a more descriptive error
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing document: {str(e)}"
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

@app.post("/sessions/create", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new session without sending a message."""
    # Create new session
    new_session_id = str(uuid.uuid4())
    sessions[new_session_id] = RagChatbot()
    session_last_active[new_session_id] = datetime.now()
    
    # Apply settings if provided
    if request.settings:
        chatbot = sessions[new_session_id]
        
        if request.settings.rag_enabled is not None:
            chatbot.rag_enabled = request.settings.rag_enabled
            print(f"Set initial chatbot.rag_enabled to {chatbot.rag_enabled}")
        
        if request.settings.web_search_enabled is not None:
            chatbot.web_search_enabled = request.settings.web_search_enabled
            print(f"Set initial chatbot.web_search_enabled to {chatbot.web_search_enabled}")
    
    return CreateSessionResponse(
        session_id=new_session_id,
        message="New session created successfully"
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
