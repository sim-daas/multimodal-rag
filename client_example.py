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

if __name__ == "__main__":
    chat_with_bot()
