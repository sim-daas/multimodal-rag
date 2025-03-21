import ollama
import base64
import os

class ImageChatbot:
    def __init__(self, model_name='llava:13b'):
        self.model_name = model_name

    def encode_image(self, image_path):
        if not os.path.isfile(image_path):
            print("Image not found. Please check the path.")
            return None
        try:
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None

    def chat_with_image(self, image_path):
        image_data = self.encode_image(image_path)
        if image_data is None:
            return "Error: Unable to load or encode image."

        print("You can ask multiple questions about the image. Type 'exit' to quit.")
        
        # Loop for multiple questions
        while True:
            question = input("Your question: ")
            if question.lower() == 'exit':
                print("Exiting the chat.")
                break
            
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": question, "images": [image_data]}
                    ]
                )
                print("Answer:", response['message']['content'])
            except Exception as e:
                print(f"Error during chat: {e}")

if __name__ == "__main__":
    image_path = input('Enter the image path: ')
    chatbot = ImageChatbot()
    chatbot.chat_with_image(image_path)
