import os
import ollama
from pathlib import Path

def process_image(image_path):
    image_path = str(image_path)
    output_dir = os.path.dirname(image_path)
    
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        res = ollama.chat(
            model='llava-llama3:latest',
            messages=[
                {'role': 'user', 'content': 
                 'Describe this image in great detail, if there is any text then extract them all in a measing full way', 'images': [image_path]}
            ]
        )
        
        description = res['message']['content']
        output_file_path = os.path.join(output_dir, f"{Path(image_path).stem}.txt")
        
        with open(output_file_path, 'w') as file:
            file.write(description)
        print(f"Description for {image_path} saved to {output_file_path}")
