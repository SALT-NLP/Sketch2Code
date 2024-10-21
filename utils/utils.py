import re
import os
import base64
import mimetypes
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from bs4 import BeautifulSoup, NavigableString
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_internvl_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def claude_encode_image(image_path: str, max_size: int = 8000):
    """Encodes an image to base64, rescaling if necessary, and determines the correct MIME type."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")

    # Open the image and resize if needed
    with Image.open(image_path) as img:
        width, height = img.size
        if width > max_size or height > max_size:
            # Calculate the new size maintaining the aspect ratio
            if width > height:
                new_height = int((max_size / width) * height)
                new_size = (max_size, new_height)
            else:
                new_width = int((max_size / height) * width)
                new_size = (new_width, max_size)

            img = img.resize(new_size, PIL.Image.LANCZOS)
            img.save("temp.png")

            with open("temp.png", "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return {"type": "base64", "media_type": mime_type, "data": encoded_string}
            os.remove("temp.png")
        else:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return {"type": "base64", "media_type": mime_type, "data": encoded_string}


def extract_html(code):
    # This regular expression finds all content between triple backticks
    matches = re.findall(r'```(.*?)```', code, re.DOTALL)
    # re.DOTALL allows the dot (.) to match newlines as well
    if matches:
        return matches[-1]  # Return the last match found
    else:
        return None


# def extract_all_questions(text):
#     # This regex finds all occurrences of the pattern 'Question:' followed by triple quotes
#     matches = re.findall(r'Question:\s*\"{3}(.*?)\"{3}', text, re.DOTALL)
#     # Each match is stripped of leading and trailing whitespace
#     return [match.strip() for match in matches] if matches else []


# def extract_all_questions(text):
#     # Regex to find all question blocks
#     blocks = re.findall(r'Question:\s*\"{3}([\s\S]*?)\"{3}', text)
#     questions = []
#     number_pattern = re.compile(r'^\d+\.\s*(.*)')  # Pattern to match 'number. ' and capture the rest
#     for block in blocks:
#         lines = block.split('\n')
#         for line in lines:
#             line = line.strip()
#             match = number_pattern.match(line)
#             if match:
#                 questions.append(match.group(1))  # Append only the question part, excluding the number
#     return questions

def extract_all_questions(text):
    # Regex to find all question blocks
    blocks = re.findall(r'Question:\s*\"{3}([\s\S]*?)\"{3}', text)
    questions = []
    number_pattern = re.compile(r'^(\d+)\.\s*(.*)')  # Adjusted pattern to capture the number and the rest of the line

    newline = '\n'
    for block in blocks:
        current_question = None
        question_buffer = []

        lines = block.split(newline)
        for line in lines:
            line = line.strip()
            match = number_pattern.match(line)
            if match:
                # If a new question starts, store the previous question if exists
                if current_question is not None:
                    questions.append(newline.join(question_buffer).strip())
                    question_buffer = []
                current_question = match.group(1)
                question_buffer.append(match.group(2))  # Start the new question's buffer with the current line's content
            elif current_question and line:  # If part of a current question and not an empty line
                question_buffer.append(line)

        # Add the last question in the block to the list, if any
        if current_question and question_buffer:
            questions.append(newline.join(question_buffer).strip())

    return questions


def encode_image(image_path):
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')

def gemini_encode_image(image_path):
    return Image.open(image_path)

def rescale_image_loader(image_path):
    """
    Load an image, rescale it so that the short side is 768 pixels.
    If after rescaling, the long side is more than 2000 pixels, return None.
    If the original short side is already shorter than 768 pixels, no rescaling is done.

    Args:
    image_path (str): The path to the image file.

    Returns:
    Image or None: The rescaled image or None if the long side exceeds 2000 pixels after rescaling.
    """
    with Image.open(image_path) as img:
        # Get original dimensions
        width, height = img.size

        # Determine the short side
        short_side = min(width, height)
        long_side = max(width, height)

        # Check if resizing is needed
        if short_side <= 768:
            if long_side > 2000:
                print ("Bad aspect ratio for GPT-4V: ", image_path)
                return None
            else:
                ## no need rescaling, return the base64 encoded image
                return encode_image(image_path)

        # Calculate new dimensions
        scaling_factor = 768 / short_side
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)

        # Check if the long side exceeds 2000 pixels after rescaling
        if new_width > 2000 or new_height > 2000:
            print ("Bad aspect ratio for GPT-4V: ", image_path)
            return None

        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        ## save to a temporary file
        resized_img = resized_img.save(image_path.replace(".png", "_rescaled.png"))
        base64_image = encode_image(image_path.replace(".png", "_rescaled.png"))
        os.remove(image_path.replace(".png", "_rescaled.png"))
        
        return base64_image

def rescale_image_loader_1(image_path):
    with Image.open(image_path) as img:
        # Get original dimensions
        width, height = img.size

        # Determine the short side
        short_side = min(width, height)
        long_side = max(width, height)

        # Check if resizing is needed
        if short_side <= 768:
            if long_side > 2000:
                print ("Bad aspect ratio for GPT-4V: ", image_path)
                return None, 0, 0
            else:
                ## no need rescaling, return the base64 encoded image
                return encode_image(image_path), width, height

        # Calculate new dimensions
        scaling_factor = 768 / short_side
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)

        # Check if the long side exceeds 2000 pixels after rescaling
        if new_width > 2000 or new_height > 2000:
            print ("Bad aspect ratio for GPT-4V: ", image_path)
            return None, 0, 0

        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        ## save to a temporary file
        resized_img = resized_img.save(image_path.replace(".png", "_rescaled.png"))
        base64_image = encode_image(image_path.replace(".png", "_rescaled.png"))
        os.remove(image_path.replace(".png", "_rescaled.png"))
        
        return base64_image, new_width, new_height

def gpt_cost(model, usage):
    '''
    Example response from GPT-4V: {'id': 'chatcmpl-8h0SZYavv8pmLGp45y05VB6NgzHxN', 'object': 'chat.completion', 'created': 1705260563, 'model': 'gpt-4-1106-vision-preview', 'usage': {'prompt_tokens': 903, 'completion_tokens': 2, 'total_tokens': 905}, 'choices': [{'message': {'role': 'assistant', 'content': '```html'}, 'finish_reason': 'length', 'index': 0}]}
    '''
    if model == "gpt-4-vision-preview":
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        cost = 0.01 * prompt_tokens / 1000 + 0.03 * completion_tokens / 1000
        return prompt_tokens, completion_tokens, cost 
    elif model == "gpt-4-1106-preview" or model == "gpt-4-1106":
        return (0.01 * usage.prompt_tokens + 0.03 * usage.completion_tokens) / 1000.0
    else:
        print ("model not supported: ", model)
        return 0


def remove_css_from_html(html_content):
    """
    Removes all CSS (contents within <style> and </style> tags) from an HTML
    webpage (provided as a string) and returns the modified HTML without CSS.

    :param html_content: A string containing the HTML content.
    :return: A string representing the HTML content without CSS.
    """
    # Using BeautifulSoup to parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Finding and removing all <style> tags along with their contents
    [style_tag.decompose() for style_tag in soup.find_all('style')]

    return str(soup)

def extract_css_from_html(html_content):
    """
    Extracts all CSS (contents within <style> and </style> tags) from an HTML
    webpage (provided as a string) and returns the CSS content.

    :param html_content: A string containing the HTML content.
    :return: A string representing the extracted CSS content.
    """
    # Using BeautifulSoup to parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extracting the CSS content from all <style> tags
    css_content = ''
    for style_tag in soup.find_all('style'):
        css_content += str(style_tag) + '\n'

    return css_content

def remove_html_comments(html_content):
    comment_pattern = r'<!.*?>'
    html_content = re.sub(comment_pattern, '', html_content, flags=re.DOTALL)
    return html_content.strip()

def extract_text_from_html(html_content):
    """
    Extracts all text elements from an HTML webpage (provided as a string)
    and returns a list of these text elements.

    :param html_content: A string containing the HTML content.
    :return: A list of strings, each representing a text element from the HTML content.
    """
    html_content = remove_html_comments(html_content)
    html_content_without_css = remove_css_from_html(html_content)
    soup = BeautifulSoup(html_content_without_css, 'html.parser')

    # Finding all text elements, excluding those within <script> tags
    texts = [element.strip().replace("\n", " ") for element in soup.find_all(string=True) if element.parent.name != 'script' and len(element.strip()) > 0 and element.strip() != 'html']
    texts = [' '.join(element.split()) for element in texts]

    return texts


def index_text_from_html(html_content):
    """
    Replaces all text elements in an HTML webpage with index markers,
    and returns the modified HTML and a dictionary mapping these indices
    to the actual text.

    :param html_content: A string containing the HTML content.
    :return: A tuple with two elements:
             1. A string of the modified HTML content.
             2. A dictionary mapping each index to the corresponding text element.
    """

    html_content = remove_html_comments(html_content)
    css = extract_css_from_html(html_content).strip()
    html_content_without_css = remove_css_from_html(html_content)
    soup = BeautifulSoup(html_content_without_css, 'html.parser')
    text_dict = {}
    index = 1

    # Iterate over each navigable string and replace it with an index
    for element in soup.find_all(string=True):
        if element.parent.name != 'script' and len(element.strip()) > 0 and element.strip() != 'html':
            clean_text = ' '.join(element.strip().replace("\n", " ").split())
            text_dict[index] = clean_text
            element.replace_with(f'[{index}]')
            index += 1
    
    html_content = str(soup)

    ## insert back css 
    head_tag = '<head>'
    end_of_head_tag_index = html_content.find(head_tag)
    if end_of_head_tag_index >= 0:
        # Calculate the position to insert the CSS (after the <head> tag)
        insert_position = end_of_head_tag_index + len(head_tag)
        # Insert the CSS string at the found position
        html_content = html_content[:insert_position] + css + "\n" + html_content[insert_position:]
        html_content = html_content.replace("<head><style>", "<head>\n<style>")
    else:
        html_content = html_content.replace("<html>", "<html>\n<head>\n</head>")
        head_tag = '<head>\n'
        end_of_head_tag_index = html_content.find(head_tag)
        end_of_head_tag_index = html_content.find(head_tag)
        insert_position = end_of_head_tag_index + len(head_tag)
        html_content = html_content[:insert_position] + css + "\n" + html_content[insert_position:]

    return html_content, text_dict


def replace_text_with_placeholder(html_content):
    """
    Replaces all text elements in an HTML webpage (provided as a string)
    with the placeholder string "placeholder" and returns the modified HTML content.

    :param html_content: A string containing the HTML content.
    :return: A string, which is the modified HTML content with text elements replaced.
    """
    css = extract_css_from_html(html_content)
    html_content_without_css = remove_css_from_html(html_content)
    soup = BeautifulSoup(html_content_without_css, 'html.parser')

    # Iterate over all text elements and replace their contents
    for element in soup.find_all(string=True):
        if element.parent.name != 'script' and len(element.strip()) > 0 and 'html' not in element.strip():
            element.replace_with("placeholder")
    
    html_content = str(soup)
    
    ## insert back css 
    head_tag = '<head>'
    end_of_head_tag_index = html_content.find(head_tag)

    # Calculate the position to insert the CSS (after the <head> tag)
    insert_position = end_of_head_tag_index + len(head_tag)
    # Insert the CSS string at the found position
    html_content = html_content[:insert_position] + css + "\n" + html_content[insert_position:]

    return html_content


def extract_title(html_content):
    """
    Extracts the title from an HTML string.

    :param html_content: A string containing HTML content.
    :return: The content of the <title> tag if found, otherwise None.
    """
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the title tag
    title_tag = soup.find('title')
    
    # Return the text of the title tag if it exists, otherwise return None
    return title_tag.text if title_tag else None

def read_html(html_path):
    with open(html_path, "r") as html_file:
        return html_file.read()


def extract_layout_description(text):
    # Regex to find all blocks marked as layout descriptions
    pattern = r'Layout Description:\s*\"{3}([\s\S]*?)\"{3}'
    descriptions = re.findall(pattern, text)
    # Return the last description if any are found, otherwise return None
    return descriptions[-1].strip() if descriptions else None

def extract_text_assignment(text):
    # Regex to find all blocks marked as layout descriptions
    pattern = r'Page Layout:\s*\"{3}([\s\S]*?)\"{3}'
    descriptions = re.findall(pattern, text)
    # Return the last description if any are found, otherwise return None
    return descriptions[-1].strip() if descriptions else None

def extract_page_layout(text):
    # Regex to find all blocks marked as layout descriptions
    pattern = r'Page Layout:\s*```([\s\S]*?)```'
    descriptions = re.findall(pattern, text)
    # Return the last description if any are found, otherwise return None
    return descriptions[-1].strip() if descriptions else None

def extract_feedback(text):
    # Regex to find all blocks marked as layout descriptions
    pattern = r'Feedback:\s*\"{3}([\s\S]*?)\"{3}'
    descriptions = re.findall(pattern, text)
    # Return the last description if any are found, otherwise return None
    return descriptions[-1].strip() if descriptions else None
    
    
def gpt_cost(model, usage):
    '''
    Example response from GPT-4V: {'id': 'chatcmpl-8h0SZYavv8pmLGp45y05VB6NgzHxN', 'object': 'chat.completion', 'created': 1705260563, 'model': 'gpt-4-1106-vision-preview', 'usage': {'prompt_tokens': 903, 'completion_tokens': 2, 'total_tokens': 905}, 'choices': [{'message': {'role': 'assistant', 'content': '```html'}, 'finish_reason': 'length', 'index': 0}]}
    '''
    if model == "gpt-4-vision-preview" or model == "gpt-4-turbo":
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        cost = 0.01 * prompt_tokens / 1000 + 0.03 * completion_tokens / 1000
        return prompt_tokens, completion_tokens, cost 
    elif model == "gpt-4-1106-preview" or model == "gpt-4-1106":
        return usage.prompt_tokens, usage.completion_tokens, (0.01 * usage.prompt_tokens + 0.03 * usage.completion_tokens) / 1000.0
    elif model == "gpt-4o":
        return usage.prompt_tokens, usage.completion_tokens, (0.0025 * usage.prompt_tokens + 0.01 * usage.completion_tokens) / 1000.0
    elif model == "gpt-4o-mini":
        return usage.prompt_tokens, usage.completion_tokens, (0.00015 * usage.prompt_tokens + 0.0006 * usage.completion_tokens) / 1000.0
    else:
        print ("model not supported: ", model)
        return 0, 0, 0

def cleanup_response(response):
    if not response:
        return None
    if '<!DOCTYPE' not in response and '<html>' not in response:
        # invalid html, return none
        return None
    ## simple post-processing
    if response[ : 3] == "```":
        response = response[3 :].strip()
    if response[-3 : ] == "```":
        response = response[ : -3].strip()
    if response[ : 4] == "html":
        response = response[4 : ].strip()

    ## strip anything before '<!DOCTYPE'
    if '<!DOCTYPE' in response:
        response = response.split('<!DOCTYPE', 1)[1]
        response = '<!DOCTYPE' + response
		
    ## strip anything after '</html>'
    if '</html>' in response:
        response = response.split('</html>')[0] + '</html>'
    return response 

def gemini_encode_image(image_path):
    return Image.open(image_path)


if __name__ == "__main__":
    reference_dir = "../../testset_100"
    predictions_dir = "../../predictions_100/gpt4v_direct_prompting"

    with open(os.path.join(reference_dir, "7713.html"), "r") as f:
        html_content = f.read()
    
    # print (extract_text_from_html(html_content))

    html, text_dict = index_text_from_html(html_content)

    with open(os.path.join(predictions_dir, "7713_temp.html"), "w") as f:
        f.write(html)
    
    print (text_dict)
