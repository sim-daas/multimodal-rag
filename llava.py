import ollama
from typing import Dict, Callable, Any

def process_image(image_url: str, question: str) -> str:
    """Process an image using LLava for multimodal reasoning."""
    response = ollama.chat(
        'llava:7b',
        messages=[
            {'role': 'user', 'content': question, 'image_url': image_url}
        ]
    )
    return response['message']['content']

# Define function calling tool for image processing
image_processing_tool = {
    'type': 'function',
    'function': {
        'name': 'process_image',
        'description': 'Process an image with a given question using LLava.',
        'parameters': {
            'type': 'object',
            'required': ['image_url', 'question'],
            'properties': {
                'image_url': {'type': 'string', 'description': 'URL of the image to process'},
                'question': {'type': 'string', 'description': 'Question about the image'},
            },
        },
    },
}

# Available function mappings
available_functions: Dict[str, Callable] = {
    'process_image': process_image,
}

def chat_with_llm(prompt: str, rag_enabled: bool = True, image_url: str = None, question: str = None):
    """Main LLM interaction with optional RAG and image processing."""
    tools = [image_processing_tool] if image_url else []
    
    messages = [{'role': 'user', 'content': prompt}]
    if image_url:
        messages.append({'role': 'user', 'content': question, 'image_url': image_url})
    
    response = ollama.chat(
        'deepseek-r1:8b',  # Replace with your chosen LLM
        messages=messages,
        tools=tools,
    )
    
    # Check for tool calls
    if response['message'].get('tool_calls'):
        for tool in response['message']['tool_calls']:
            function_to_call = available_functions.get(tool['function']['name'])
            if function_to_call:
                print(f"Calling function: {tool['function']['name']}")
                output = function_to_call(**tool['function']['arguments'])
                print(f"Function output: {output}")
                return output
    
    return response['message']['content']

# Example Usage
if __name__ == "__main__":
    prompt = "Describe the content of this image."
    image_url = "https://example.com/sample.jpg"  # Replace with actual image
    question = "What objects are present?"
    
    response = chat_with_llm(prompt, rag_enabled=True, image_url=image_url, question=question)
    print("Response:", response)
