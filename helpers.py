import os
from base64 import b64encode
import PIL
import PIL.Image
from utils import *
import shutil
import streamlit as st
# Threshold to avoid smaller images
MIN_IMAGE_HEIGHT = 50
MIN_IMAGE_WIDTH = 50

def summarize_prompt(element):
    '''
        Creates a prompt for summarizing a text or table element.

        Parameters:
        element (str): The text or table element to be summarized.

        Returns:
        str: A formatted prompt for generating a summary.
    '''
    prompt = f"""
        You are an assistant tasked with summarizing tables and text for retrieval.
        These summaries will be embedded and used to retrieve the raw text and table 
        elements. Give a concise summary of the table or text that is well-optimized 
        for retrieval. Table or text: {element}"""
    return prompt

def generate_summaries(model, texts, tables, summarize_texts=False):
    '''
        Generates summaries for texts and tables using a given model.

        Parameters:
        model: The model used to generate the summaries.
        texts (list): List of text data.
        tables (list): List of table data.
        summarize_texts (bool): Flag to indicate if texts should be summarized. Default is False.

        Returns:
        list: Summaries of the texts.
        list: Summaries of the tables.
    '''
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

            if response and response.candidates:
                candidate = response.candidates[0]
                if len(candidate.content.parts)>0:
                    res = candidate.content.parts[0].text
                    if res:
                        table_summaries.append(res)
                    else:
                        table_summaries.append(str(table))
                else:
                        table_summaries.append(str(table))
            else:
                table_summaries.append(str(table))
    return text_summaries, table_summaries

def generate_image_summaries(model, folder_id):
    '''
        Generates summaries for images in a specified folder using a given model and uploads them to the cloud.

        Parameters:
        model: The model used to generate the image summaries.
        folder_id (str): ID of the folder containing the extracted images.

        Returns:
        list: Summaries of the images.
        list: Cloudinary URLs of the uploaded images.
    '''
    image_summaries = []
    image_filepaths=[]
    image_dir = os.path.join('figures', folder_id)
    prompt = """
        You are an automotive assistant tasked with summarizing images for retrieval.
        These summaries will be embedded and used to retrieve the raw image. Describe 
        conciesly the characters (shape, color), and infer a little what the image means.
        If the image is a table or has a table, extract all data from it and summarize it.
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
                if response and response.candidates:
                    candidate = response.candidates[0]
                    if len(candidate.content.parts)>0:
                        res = candidate.content.parts[0].text
                        image_summaries.append(res)
                        # upload image to cloudinary and get url
                        url = upload_image(image_path)
                        image_filepaths.append(url)
    shutil.rmtree(image_dir)
    return image_summaries, image_filepaths

def load_image_as_base64(file_path):
    '''
        Loads an image file and encodes it as a base64 string.

        Parameters:
        file_path (str): Path to the image file.

        Returns:
        str: Base64 encoded string of the image.
    '''
    with open(file_path, "rb") as image_file:
        return b64encode(image_file.read()).decode("utf-8")
    
def summarize_image(model, image_path):
    '''
        Generates a summary for an image using a given model.

        Parameters:
        model: The model used to generate the image summary.
        image_path (str): Path to the image file.

        Returns:
        str: The generated image summary.
    '''
    prompt = """
        You are an automotive assistant tasked with summarizing images for retrieval.
        These summaries will be embedded and used to retrieve the raw image. Describe 
        conciesly the characters (shape, color), and infer a little what the image means.
        If the image is a table or has a table, extract data from the table and summarize it.
    """
    summary=""
    with PIL.Image.open(image_path) as img:
        response = model.generate_content([prompt, img])
    
        if response and hasattr(response, 'text'):
            summary += response.text
        else:
            summary += "Sorry could not analyse the image. Try again"
    os.remove(image_path)
    return summary

def get_car_name(model, texts):
    '''
        Extracts and returns the name of a car from the provided textual context using a given model.

        Parameters:
        model: The model used to generate the car name.
        texts (list of str): List of text snippets containing the context from which to extract the car name.

        Returns:
        str: The name of the car in uppercase if successfully extracted, otherwise "Unknown".
    '''
    content=""
    for text in texts:
        content += text

    prompt = f"""
        You are an automotive assistant tasked with naming ONLY THE NAME OF THE CAR, \
        based on the CONTEXT provided. If you cannot find a car name, respond with UNKNOWN. \n: CONTEXT: {content}"""
    
    try:
        response = model.generate_content(prompt)

        if response and hasattr(response, 'text'):
            response = response.text.upper()
    except:
        st.toast("Error getting Car name.")
        response = "UNKNOWN"
        return response
    return response