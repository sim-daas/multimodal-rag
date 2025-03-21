#!/usr/bin/env python

import requests
import json
import os

# API base URL
BASE_URL = "http://localhost:8000"

def print_json(data):
    """Print JSON data in a readable format."""
    print(json.dumps(data, indent=2))

def chat_with_bot():
    """Example of chatting with the bot through the API."""
    session_id = None
    
    print("=== RAG Chatbot API Client Example ===")
    print("Type 'quit' to exit, 'commands' to see available commands")
    
    while True:
        user_input = input("> ")
        
        if user_input.lower() == "quit":
            break
        
        if user_input.lower() == "commands":
            print("\nAvailable commands:")
            print("  toggle_rag on/off - Enable or disable RAG retrieval")
            print("  toggle_websearch on/off - Enable or disable web search")
            print("  save_conversation <filename> - Save the current conversation")
            print("  list_conversations - List all saved conversations")
            print("  load_conversation <name> - Load a saved conversation")
            print("  session_info - Show information about the current session")
            print("  import_document <filepath> - Import a document or a list of documents")
            print("  process_audio <filepath> - Transcribe and chat with audio file")
            print("  transcribe_audio <filepath> - Just transcribe audio without chat")
            print("  quit - Exit the chat")
            continue
        
        if user_input.lower().startswith("toggle_rag"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1] in ["on", "off"]:
                enabled = parts[1] == "on"
                toggle_feature(session_id, "rag", enabled)
            else:
                print("Usage: toggle_rag on/off")
            continue
            
        if user_input.lower().startswith("toggle_websearch"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1] in ["on", "off"]:
                enabled = parts[1] == "on"
                toggle_feature(session_id, "websearch", enabled)
            else:
                print("Usage: toggle_websearch on/off")
            continue
        
        if user_input.lower().startswith("save_conversation"):
            parts = user_input.split()
            if len(parts) == 2:
                save_conversation(session_id, parts[1])
            else:
                print("Usage: save_conversation <filename>")
            continue
            
        if user_input.lower() == "list_conversations":
            list_conversations()
            continue
            
        if user_input.lower().startswith("load_conversation"):
            parts = user_input.split()
            if len(parts) == 2:
                new_session_id = load_conversation(parts[1])
                if new_session_id:
                    session_id = new_session_id
            else:
                print("Usage: load_conversation <name>")
            continue
            
        if user_input.lower() == "session_info":
            get_session_info(session_id)
            continue
            
        if user_input.lower().startswith("import_document"):
            parts = user_input.split()
            if len(parts) == 2:
                if not session_id:
                    print("No active session. Send a message first.")
                else:
                    import_document(session_id, parts[1])
            else:
                print("Usage: import_document <filepath>")
            continue
            
        if user_input.lower().startswith("process_audio"):
            parts = user_input.split()
            if len(parts) == 2:
                if process_audio_chat(parts[1], session_id):
                    session_id = process_audio_chat(parts[1], session_id)
            else:
                print("Usage: process_audio <filepath>")
            continue

        if user_input.lower().startswith("transcribe_audio"):
            parts = user_input.split()
            if len(parts) == 2:
                transcribe_audio(parts[1], session_id)
            else:
                print("Usage: transcribe_audio <filepath>")
            continue
        
        # Regular chat message
        response = send_message(user_input, session_id)
        
        if response:
            print(f"\nAI: {response['message']}")
            # Update session ID if we got a new one
            session_id = response["session_id"]

def send_message(message, session_id=None):
    """Send a message to the chatbot."""
    try:
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id
            
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error sending message: {e}")
        return None

def toggle_feature(session_id, feature, enabled):
    """Toggle a feature (rag or websearch)."""
    if not session_id:
        print("No active session. Send a message first.")
        return
        
    try:
        payload = {
            "session_id": session_id,
            "feature": feature,
            "enabled": enabled
        }
        
        response = requests.post(f"{BASE_URL}/features/toggle", json=payload)
        response.raise_for_status()
        result = response.json()
        
        print(result["message"])
    except requests.RequestException as e:
        print(f"Error toggling feature: {e}")

def save_conversation(session_id, filename):
    """Save the current conversation."""
    if not session_id:
        print("No active session. Send a message first.")
        return
        
    try:
        payload = {
            "session_id": session_id,
            "filename": filename
        }
        
        response = requests.post(f"{BASE_URL}/conversations/save", json=payload)
        response.raise_for_status()
        result = response.json()
        
        print(result["message"])
    except requests.RequestException as e:
        print(f"Error saving conversation: {e}")

def list_conversations():
    """List all saved conversations."""
    try:
        response = requests.get(f"{BASE_URL}/conversations")
        response.raise_for_status()
        result = response.json()
        
        print("\nSaved conversations:")
        for conv in result["conversations"]:
            print(f"  - {conv}")
    except requests.RequestException as e:
        print(f"Error listing conversations: {e}")

def load_conversation(name):
    """Load a saved conversation."""
    try:
        payload = {
            "conversation_name": name
        }
        
        response = requests.post(f"{BASE_URL}/conversations/load", json=payload)
        response.raise_for_status()
        result = response.json()
        
        print(result["message"])
        return result["session_id"]
    except requests.RequestException as e:
        print(f"Error loading conversation: {e}")
        return None

def import_document(session_id, file_path):
    """Import a document or list of documents into the system."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    # Check if the file is a text file that might contain a list of documents
    if file_path.lower().endswith('.txt'):
        # Check first line to determine if it's a list file
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                # If first line is a path or starts with #, treat as a document list
                if os.path.exists(first_line) or first_line.startswith('#'):
                    process_document_list(session_id, file_path)
                    return
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            # If there's an error reading the file, try uploading as a regular document
            pass
    
    # If not determined to be a list file, upload as a single document
    upload_single_document(session_id, file_path)

def upload_single_document(session_id, file_path):
    """Upload a single document file to the API."""
    try:
        print(f"Uploading document: {file_path}")
        
        # Create multipart form data with the file and session_id
        with open(file_path, 'rb') as file:
            files = {'file': (os.path.basename(file_path), file)}
            data = {'session_id': session_id}
            
            response = requests.post(
                f"{BASE_URL}/documents/upload",
                files=files,
                data=data
            )
            
            response.raise_for_status()
            result = response.json()
            
            print(f"Upload successful: {result['message']}")
    except Exception as e:
        print(f"Error uploading document: {str(e)}")

def process_document_list(session_id, list_file_path):
    """Process a list of documents from a file."""
    try:
        print(f"Processing document list from: {list_file_path}")
        
        # Read the file to check if it contains valid paths
        document_paths = []
        with open(list_file_path, 'r') as f:
            for line in f:
                path = line.strip()
                if path and not path.startswith('#'):  # Skip empty lines and comments
                    document_paths.append(path)
        
        if not document_paths:
            print("No valid document paths found in the list file")
            return
            
        # Check if the server has direct access to these paths
        print(f"Found {len(document_paths)} documents in list")
        print("Option 1: Send list file path to server (if server has access to these paths)")
        print("Option 2: Upload each document individually")
        choice = input("Enter 1 or 2: ")
        
        if choice == "1":
            # Send the list file path to the server for processing
            payload = {
                "session_id": session_id,
                "document_list_path": list_file_path
            }
            
            response = requests.post(
                f"{BASE_URL}/documents/process-list",
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            
            print(f"Document processing: {result['message']}")
            print(f"Processed {result.get('processed', 0)} documents successfully, {result.get('failed', 0)} failed")
            
        elif choice == "2":
            # Upload each document individually
            successful = 0
            failed = 0
            
            for i, doc_path in enumerate(document_paths):
                print(f"Uploading document {i+1}/{len(document_paths)}: {doc_path}")
                if os.path.exists(doc_path):
                    try:
                        with open(doc_path, 'rb') as file:
                            files = {'file': (os.path.basename(doc_path), file)}
                            data = {'session_id': session_id}
                            
                            response = requests.post(
                                f"{BASE_URL}/documents/upload",
                                files=files,
                                data=data
                            )
                            
                            response.raise_for_status()
                            successful += 1
                    except Exception as e:
                        print(f"  Error uploading {doc_path}: {str(e)}")
                        failed += 1
                else:
                    print(f"  File not found: {doc_path}")
                    failed += 1
            
            print(f"Completed batch upload: {successful} successful, {failed} failed")
        else:
            print("Invalid choice")
            
    except Exception as e:
        print(f"Error processing document list: {str(e)}")

def get_session_info(session_id):
    """Get information about the current session."""
    if not session_id:
        print("No active session. Send a message first.")
        return
        
    try:
        response = requests.get(f"{BASE_URL}/sessions/{session_id}")
        response.raise_for_status()
        info = response.json()
        
        print("\nSession Information:")
        print(f"  Session ID: {info['session_id']}")
        print(f"  Last active: {info['last_active']}")
        print(f"  RAG enabled: {info['rag_enabled']}")
        print(f"  Web search enabled: {info['web_search_enabled']}")
        print(f"  Message count: {info['message_count']}")
    except requests.RequestException as e:
        print(f"Error getting session information: {e}")

def process_audio_chat(file_path, session_id=None):
    """Process an audio file and chat about its content."""
    if not os.path.exists(file_path):
        print(f"Error: Audio file not found: {file_path}")
        return session_id
    
    try:
        print(f"Processing audio file: {file_path}")
        print("Uploading and transcribing audio...")
        
        # Create multipart form data with the file and session_id
        with open(file_path, 'rb') as file:
            files = {'file': (os.path.basename(file_path), file)}
            data = {}
            if session_id:
                data['session_id'] = session_id
            
            response = requests.post(
                f"{BASE_URL}/audio/chat",
                files=files,
                data=data
            )
            
            response.raise_for_status()
            result = response.json()
            
            print(f"\nTranscribed text: {result.get('transcribed_text', 'Transcription not returned')}")
            print(f"\nAI: {result['message']}")
            
            return result["session_id"]
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return session_id

def transcribe_audio(file_path, session_id=None):
    """Transcribe an audio file without generating a chat response."""
    if not os.path.exists(file_path):
        print(f"Error: Audio file not found: {file_path}")
        return
    
    try:
        print(f"Transcribing audio file: {file_path}")
        
        # Create multipart form data with the file and session_id
        with open(file_path, 'rb') as file:
            files = {'file': (os.path.basename(file_path), file)}
            data = {'add_to_conversation': 'false'}
            if session_id:
                data['session_id'] = session_id
            
            response = requests.post(
                f"{BASE_URL}/audio/transcribe",
                files=files,
                data=data
            )
            
            response.raise_for_status()
            result = response.json()
            
            if result["success"]:
                print(f"\nTranscribed text: {result['text']}")
                if result.get('duration'):
                    print(f"Audio duration: {result['duration']} seconds")
                if result.get('language'):
                    print(f"Detected language: {result['language']}")
            else:
                print(f"\nTranscription failed: {result.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")

if __name__ == "__main__":
    chat_with_bot()
