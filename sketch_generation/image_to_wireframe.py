from utils.screenshot import take_screenshot, take_screenshot_from_html

# import pytesseract
from paddleocr import PaddleOCR, draw_ocr

from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

import numpy as np
import torch
import math
import random
import os
import argparse
import cv2


"""
nlprun -m sphinx2 -g 1 -c 4 -r 40G -a sketch2code --output /sailhome/lansong/Sketch2Code/logs/generated_sketch_v0.3.txt 'python /sailhome/lansong/Sketch2Code/Design2Code/data_utils/sketch2code/image_to_wireframe.py'
"""


ocr = PaddleOCR(use_gpu=True, use_angle_cls=True, lang='en', show_log=False)

# def do_ocr(img):
#     ocr = PaddleOCR(use_gpu=False, use_angle_cls=True, lang='en', show_log=False)
    
#     result = ocr.ocr(img, cls=True)
#     return result


def convert_to_canny(img, low_threshold=50, high_threshold=150):
    """
    Convert an image to a Canny edge image.
    :param image: Input image.
    :param low_threshold: Lower threshold for the hysteresis procedure.
    :param high_threshold: Higher threshold for the hysteresis procedure.
    """
    # Apply Canny Edge Detection
    canny_img = cv2.Canny(img, low_threshold, high_threshold)

    return canny_img

def bezier_curve(points, t):
    # De Casteljau's algorithm for Bezier curves
    while len(points) > 1:
        new_points = []
        for i in range(len(points) - 1):
            new_points.append((1 - t) * np.array(points[i]) + t * np.array(points[i + 1]))
        points = new_points
    return points[0]

def draw_wavy_line(draw, start_point, end_point, box_height, base_wave_length=10):
    stroke_width = max(1, int(box_height * 0.15))  # Stroke width as 15% of box height
    # wave_height = max(1, int(box_height * 0.4))
    wave_height = max(10, int(box_height * 0.4))

    # Generate intermediate control points for Bezier curve
    box_width = np.linalg.norm(np.array(start_point) - np.array(end_point))
    num_waves = max(2, int(box_width / base_wave_length))
    step = (np.array(end_point) - np.array(start_point)) / (num_waves + 1)
    all_points = [start_point]
    for i in range(1, num_waves + 1):
        point = np.array(start_point) + step * i
        randx = random.uniform(-0.2, 0.4)
        randy = random.uniform(-step[0] * 0.4, step[0] * 0.4)
        point[1] = point[1] + wave_height * (1 + randy) if i % 2 == 0 else point[1] - wave_height * (1 + randy)
        point[0] += base_wave_length * randx
        
        all_points.append(tuple(point))
    all_points.append(end_point)

    # Compute the curve with all control points
    curve_points = [bezier_curve(all_points, t) for t in np.linspace(0, 1, 100)]
    curve_points = [(int(x), int(y)) for x, y in curve_points]

    # Draw the wavy line
    draw.line(curve_points, fill=(255), width=stroke_width)


def draw_sine_wave(draw, start_point, end_point, box_height, base_wave_length=30):
    stroke_width = max(1, int(box_height * 0.15))  # Stroke width as 15% of box height
    wave_height = min(10, int(box_height * 0.5))

    # Compute the length of the line
    start_x, start_y = start_point
    end_x, end_y = end_point
    length = end_x - start_x

    # Prepare to draw the sine wave along the horizontal line
    num_points = 100  # Number of points to draw the sine wave
    x_values = np.linspace(start_x, end_x, num_points)
    # Frequency adjustment to ensure waves are of length 'base_wave_length'
    frequency = (2 * math.pi) / base_wave_length
    y_values = start_y + wave_height * np.sin(frequency * (x_values - start_x))

    # Create the list of points for the sine wave
    curve_points = [(int(x), int(y)) for x, y in zip(x_values, y_values)]

    # Draw the sine wave
    draw.line(curve_points, fill=(255), width=stroke_width)



def draw_text_template(img, result):
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    for line in result:
        if not line:
            continue
        for box in line:
            # print(box)
            coords = box[0]  # The bounding box vertices list
            text, confidence = box[1]  # Text detected and confidence score
            (pt1, pt2, pt3, pt4) = coords
            box_height = height = min(abs(pt4[1] - pt1[1]), abs(pt3[1] - pt2[1]))
            
            start = [max(pt4[0], pt1[0]), int((pt4[1]+pt1[1])/2)]
            end = [min(pt2[0], pt3[0]), int((pt2[1]+pt3[1])/2)]
            draw_wavy_line(draw, start, end, box_height)
            
            # draw_wavy_line(draw, start, end, box_height, 6)
            # draw_sine_wave(draw, start, end, box_height)
    
    res_img = np.array(pil_img)
    if res_img.shape[-1] == 4:  # Check for RGBA and convert to RGB
        res_img = cv2.cvtColor(res_img, cv2.COLOR_RGBA2RGB)
    
    return res_img



def remove_text(img, result):
    """
    Detects text in an image and replaces it with placeholder rectangles.
    :param img: Input image.
    :param result: OCR results.
    """
    
    # Loop through the detected text boxes
    for line in result:
        if not line:
            continue
        for box in line:
            # print(box)
            coords = box[0]  # The bounding box vertices list
            text, confidence = box[1]  # Text detected and confidence score
            # Draw rectangles over the detected text regions
            pts = np.array(coords, dtype=np.int32)
            cv2.fillPoly(img, [pts], (0, 0, 0))  # Fill the polygon
    return img

def convert_to_sketch(img):
    # img = cv2.imread(image_path)
    img_np = np.array(img)
    if img_np.shape[-1] == 4:  # Check for RGBA and convert to RGB
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = img_np
    
    ocr_results = ocr.ocr(img_rgb, cls=True)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    sketch = convert_to_canny(gray)
    sketch = remove_text(sketch, ocr_results)
    sketch = draw_text_template(sketch, ocr_results)
    sketch = 255 - sketch
    return sketch
    
def process_images(data, out_dir, num_samples=100, seed=1234):
    random.seed(seed)
    data = data.shuffle(seed=seed).select(range(num_samples))
    
    for i, example in tqdm(enumerate(data), total=len(data)):
        img = example['image']
        # print(img)
        output_path = os.path.join(out_dir, "curly_line", f'{i}.png')
        sketch = convert_to_sketch(img)
        cv2.imwrite(output_path, sketch)


def read_html(html_path):
    with open(html_path, "r") as html_file:
        return html_file.read()

def process_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Replace the src in all img tags and remove any inline style src setting
    for img in soup.find_all('img'):
        img['src'] = 'placeholder3.png'
        if 'style' in img.attrs and 'src' in img['style']:
            styles = img['style'].split(';')
            new_styles = [style for style in styles if not style.strip().startswith('src')]
            img['style'] = ';'.join(new_styles)
    return soup.prettify()

def process_directory(data_dir, out_dir, debug=False):
    all_files = os.listdir(data_dir)
    if args.sanity_check:
        all_files = all_files[:5]
        print(all_files)
    
    for filename in tqdm(all_files):
        if filename.endswith('.html'):
            img_id = filename.split('.')[0]
            html_path = os.path.join(data_dir, filename)
            html = process_html(read_html(html_path))
            temp_path = os.path.join(out_dir, "tmp.html")
            with open(temp_path, 'w') as f:
                f.write(html)
            # img = take_screenshot_from_html(html)
            img = take_screenshot(temp_path)
            sketch = convert_to_sketch(img)
            
            output_path = os.path.join(out_dir, f"{img_id}_out.png")
            cv2.imwrite(output_path, sketch)

        

if __name__ == "__main__":
    print("hello world!!")
    # dataset = load_dataset("/juice2/scr2/nlp/pix2code/zyanzhe/WebSight_82k_train/", split='train[:100]')
    # dataset = load_from_disk("/juice2/scr2/nlp/pix2code/zyanzhe/WebSight_82k_train/")
    # print(dataset)
    # out_dir = "/sailhome/lansong/Sketch2Code/cv2_sketch"

    # process_images(dataset, out_dir)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_dataset_v1')
    parser.add_argument('--out_dir', type=str, default='/sailhome/lansong/Sketch2Code/sketch2code_generated_sketch')
    parser.add_argument("--sanity_check", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    debug = args.sanity_check
    data_dir = args.input_dir
    out_dir = args.out_dir
    process_directory(data_dir, out_dir, debug)