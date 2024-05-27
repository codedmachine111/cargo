import os
from base64 import b64encode
import base64
import PIL
import PIL.Image
from utils import *

MIN_IMAGE_HEIGHT = 50
MIN_IMAGE_WIDTH = 50

# Generate summarization prompt
def summarize_prompt(element):
    prompt = f"""
        You are an assistant tasked with summarizing tables and text for retrieval.
        These summaries will be embedded and used to retrieve the raw text and table 
        elements. Give a concise summary of the table or text that is well-optimized 
        for retrieval. Table or text: {element}"""
    return prompt

# Generate summaries
def generate_summaries(model, texts, tables, summarize_texts=False):
    text_summaries=[]
    table_summaries=[]

    if texts:
        if summarize_texts:
            for text in texts:
                prompt = summarize_prompt(text)
                response = model.generate_content(prompt)
                text_summaries.append(response.text)
        else:
            text_summaries = texts
    if tables:
        for table in tables:
            prompt = summarize_prompt(table)
            response = model.generate_content(prompt)
            if response and hasattr(response, 'text'):
                table_summaries.append(response.text)
            else:
                table_summaries.append("Summary not available")

    return text_summaries, table_summaries

# Encode image to base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Generate summaries for all images
def generate_image_summaries(model, folder_id):
    image_summaries = []
    image_filepaths=[]
    image_dir = os.path.join('figures', folder_id)
    prompt = """
        You are an automotive assistant tasked with summarizing images for retrieval.
        These summaries will be embedded and used to retrieve the raw image. Describe 
        conciesly the characters (shape, color), and infer a little what the image means.
    """

    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)

            with PIL.Image.open(image_path) as img:
                width, height = img.size

                # Skip small images
                if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                    continue
                
                response = model.generate_content([prompt, img])
                if response and hasattr(response, 'text'):
                    image_summaries.append(response.text)
                    # upload image to cloudinary and get url
                    url = upload_image(image_path)
                    image_filepaths.append(url)

    return image_summaries, image_filepaths


def load_image_as_base64(file_path):
    with open(file_path, "rb") as image_file:
        return b64encode(image_file.read()).decode("utf-8")