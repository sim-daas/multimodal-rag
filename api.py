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

from newclass import RagChatbot, VectorStore, WebSearch, ConversationManager

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
        
        # Add document to vector store
        with open(temp_file_path, "r", encoding="utf-8", errors="replace") as f:
            document_content = f.read()
        
        chatbot.vectorstore.add_document(document_content, source=file.filename)
        
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
