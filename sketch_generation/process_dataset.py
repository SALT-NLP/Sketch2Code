import cv2
import numpy as np
import os
import random
import io

from bs4 import BeautifulSoup
from PIL import Image
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from utils.screenshot import take_screenshot, take_screenshot_from_html
from image_to_wireframe import convert_to_sketch


"""
nlprun -m jagupard30 -g 1 -c 4 -r 80G -a sketch2code -p high --output /sailhome/lansong/Sketch2Code/logs/image_to_sketchv0.2.txt 'python /sailhome/lansong/Sketch2Code/Design2Code/data_utils/sketch2code/clean_html_for_sketch.py'

"""




def convert_to_base64(img):
    """Convert an image to a base64 string."""
    img = Image.fromarray(img)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def process_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Add the border style to the beginning of the <style> tag
    # style_tag = soup.find('style')
    # if style_tag:
    #     # Prepend the new style for img
    #     style_tag.string = "img {border: 2px solid black;}" + style_tag.string
    # else:
    #     # If no style tag exists, create one
    #     new_style_tag = soup.new_tag("style")
    #     new_style_tag.string = "img {border: 2px solid black;}"
    #     # soup.head.append(new_style_tag)
    #     soup.html.insert(0, new_style_tag)

    # Replace the src in all img tags and remove any inline style src setting
    for img in soup.find_all('img'):
        img['src'] = 'placeholder3.png'
        if 'style' in img.attrs and 'src' in img['style']:
            styles = img['style'].split(';')
            new_styles = [style for style in styles if not style.strip().startswith('src')]
            img['style'] = ';'.join(new_styles)
    return soup.prettify()


def process_example(example):
    html = example['text']
    html = process_html(html)
    img = take_screenshot_from_html(html)
    sketch = convert_to_sketch(img)
    # cv2.imwrite(f"/sailhome/lansong/Sketch2Code/cv2_sketch/temp/test.png", sketch)
    # print(sketch)
    # sketch_base64 = convert_to_base64(sketch)
    return {'html_clean': html, 'sketch': sketch}


def process_htmls(data, out_dir, num_samples=100, seed=1234):
    random.seed(seed)
    data = data.shuffle(seed=seed).select(range(num_samples))
    
    for i, example in tqdm(enumerate(data), total=len(data)):
        html = example['text']
        code_path = os.path.join(out_dir, "processed", f'{i}.html')
        output_path = os.path.join(out_dir, "processed", f'{i}.png')
        html = process_html(html)
        with open(code_path, 'w') as f:
            f.write(html)
        img = take_screenshot(code_path)
        sketch = convert_to_sketch(img)
        cv2.imwrite(output_path, sketch)


def process_dataset(data, out_dir, num_samples=-1, seed=1234, debug=False):
    if num_samples > 0:
        random.seed(seed)
        data = data.shuffle(seed=seed).select(range(num_samples))
    
    updated_dataset = data.map(process_example, batched=False)
    updated_dataset.save_to_disk(out_dir)
    
    if debug:
        for i, exp in tqdm(enumerate(updated_dataset), total=len(updated_dataset)):
            # print(exp["sketch"])
            cv2.imwrite(f"/sailhome/lansong/Sketch2Code/cv2_sketch/temp/{i}.png", np.asarray(exp["sketch"], dtype=np.uint8))
            # img.save(f"/sailhome/lansong/Sketch2Code/cv2_sketch/temp/{i}.png")
        
            html_cleaned = exp["html_clean"]
            with open(f"/sailhome/lansong/Sketch2Code/cv2_sketch/temp/{i}.html", "w") as f:
                f.write(html_cleaned)
                
        dataset = load_from_disk(out_dir)
        print(dataset)
        for i, exp in tqdm(enumerate(dataset), total=len(dataset)):
            cv2.imwrite(f"/sailhome/lansong/Sketch2Code/cv2_sketch/temp/{i}-1.png", np.asarray(exp["sketch"], dtype=np.uint8))
            
            html_cleaned = exp["html_clean"]
            with open(f"/sailhome/lansong/Sketch2Code/cv2_sketch/temp/{i}-1.html", "w") as f:
                f.write(html_cleaned)
            
            



if __name__ == "__main__":
    # dataset = load_dataset("/juice2/scr2/nlp/pix2code/zyanzhe/WebSight_82k_train/", split='train[:100]')
    dataset = load_from_disk("/juice2/scr2/nlp/pix2code/zyanzhe/WebSight_82k_train/")
    print(dataset)
    
    out_dir = "/juice2/scr2/nlp/pix2code/zyanzhe/Sketch2Code_82k_train_v0/"

    process_dataset(dataset, out_dir)
    
    # dataset = load_from_disk(out_dir)
    # print(dataset)
    # for i, exp in tqdm(enumerate(dataset), total=len(dataset)):
    #     print(exp["sketch"])
    #     img = Image.fromarray(np.asarray(exp["sketch"], dtype=np.int32))
    #     img.save(f"/sailhome/lansong/Sketch2Code/cv2_sketch/temp/{i}.png")
        
    #     html_cleaned = exp["html_clean"]
    #     with open(f"/sailhome/lansong/Sketch2Code/cv2_sketch/temp/{i}.html", "w") as f:
    #         f.write(html_cleaned)
    