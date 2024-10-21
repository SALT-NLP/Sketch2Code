import cv2
# import pytesseract
from paddleocr import PaddleOCR, draw_ocr

from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset, load_from_disk
import numpy as np
import torch
import random
import os



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

def apply_random_distortion(img, result, intensity=2, randomness=1):
    # First, black out the text areas so that they won't be affected by the distortions
    text_mask = np.zeros_like(img, dtype=np.uint8)
    edges = np.array(img, copy=True)
    
    for line in result:
        if not line:
            continue
        for box in line:
            coords = box[0]  # The bounding box vertices list
            text, confidence = box[1]
            pts = np.array(coords, dtype=np.int32)
            cv2.fillPoly(text_mask, [pts], (255))
            cv2.fillPoly(edges, [pts], (0, 0, 0))
    
    # Create a coordinate grid
    xx, yy = np.meshgrid(np.arange(edges.shape[1]), np.arange(edges.shape[0]))
    
    # Create random displacement fields
    dx = (np.random.rand(*edges.shape) * 2 - 1) * randomness
    dy = (np.random.rand(*edges.shape) * 2 - 1) * randomness
    
    # Apply the distortions to the grid
    map_x = (xx + dx).astype(np.float32)
    map_y = (yy + dy).astype(np.float32)

    # Remap the edges using the distorted grid
    distorted = cv2.remap(edges, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    # Dilate the result to enhance visibility
    kernel = np.ones((intensity, intensity), np.uint8)
    distorted = cv2.dilate(distorted, kernel, iterations=1)
    
    
    return np.where(text_mask==255, img, distorted)


def fit_text_to_box(font_path, text, box_width, box_height):
    # Start with a default font size
    font_size = 1
    font = ImageFont.truetype(font_path, font_size)
    
    # Increase font size until the text width is less than the bounding box width
    while font.getsize(text)[0] < box_width and font.getsize(text)[1] < box_height:
        font_size += 1
        font = ImageFont.truetype(font_path, font_size)
    
    # Adjust back one step when it exceeds the width
    if font.getsize(text)[0] > box_width or font.getsize(text)[1] > box_height:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)

    return font


def do_ocr(img):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    result = ocr.ocr(img, cls=True)
    return result


def replace_text_with_boxes(img, result):
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
            cv2.fillPoly(img, [pts], (255, 255, 255))  # Fill the polygon
    
    
    # boxes = pytesseract.image_to_boxes(Image.fromarray(img))

    # # Draw rectangles over detected text areas
    # for b in boxes.splitlines():
    #     b = b.split(' ')
    #     img = cv2.rectangle(img, (int(b[1]), img.shape[0] - int(b[2])), (int(b[3]), img.shape[0] - int(b[4])), (255, 255, 255), -1)

    return img


def copy_text_to_canny(gray, sketch, result):    
    text_mask = np.zeros_like(gray, dtype=np.uint8)
    
    for line in result:
        if not line:
            continue
        for box in line:
            coords = box[0]  # The bounding box vertices list
            text, confidence = box[1]
            pts = np.array(coords, dtype=np.int32)
            cv2.fillPoly(text_mask, [pts], (255))
    
    return np.where(text_mask==255, gray, sketch)


def print_to_handwritten(img, result, font_path, header_only=False):
    # first, convert the box area to black
    for line in result:
        if not line:
            continue
        for box in line:
            coords = box[0]  # The bounding box vertices list
            text, _ = box[1]
            
            pts = np.array(coords, dtype=np.int32)
            words = text.split()
            # if header_only and len(words.split()) > 3:
            #     # not a header, replace with white box placeholder
            cv2.fillPoly(img, [pts], (0, 0, 0))
    
    # load the pil image
    pil_img = Image.fromarray(img)
    # Create ImageDraw object
    draw = ImageDraw.Draw(pil_img)  
    
    # Redraw the image with the text in a dynamically sized handwriting font
    for line in result:
        if not line:
            continue
        for box in line:
            coords = box[0]  # The bounding box vertices list
            text, _ = box[1]
            # now add the handwritten text
            (pt1, pt2, pt3, pt4) = coords
            width = min(abs(pt2[0] - pt1[0]), abs(pt3[0] - pt4[0]))
            height = min(abs(pt4[1] - pt1[1]), abs(pt3[1] - pt2[1]))
            font = fit_text_to_box(font_path, text, width, height)
            x, y = pt1
            draw.text((x, y), text, font=font, fill=255)
    
    res_img = np.array(pil_img)
    if res_img.shape[-1] == 4:  # Check for RGBA and convert to RGB
        res_img = cv2.cvtColor(res_img, cv2.COLOR_RGBA2RGB)
    
    return res_img


def convert_to_sketch(img, output_path):
    # img = cv2.imread(image_path)
    img_np = np.array(img)
    if img_np.shape[-1] == 4:  # Check for RGBA and convert to RGB
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = img_np
    
    font_path = "/sailhome/lansong/Sketch2Code/fonts/Zeyada-Regular.ttf"
    ocr_results = do_ocr(img_rgb)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    # sketch = convert_to_canny(img_rgb)
    sketch = convert_to_canny(gray)
    sketch = print_to_handwritten(sketch, ocr_results, font_path, header_only=True)
    
    # sketch = copy_text_to_canny(255 - gray, sketch, ocr_results)
    # sketch = replace_text_with_boxes(sketch, ocr_results)
    # sketch = apply_random_distortion(sketch, ocr_results)
    
    
    
    # contours, _ = cv2.findContours(sketch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # # Create a blank white canvas the same size as the original image
    # canvas = np.ones(img_rgb.shape, dtype=np.uint8) * 255
    
    # # Draw the contours on the canvas to create the sketch effect
    # cv2.drawContours(canvas, contours, -1, (0, 0, 0), 1)

    # invert the color
    sketch = 255 - sketch
    
    cv2.imwrite(output_path, sketch)
    # cv2.imwrite(output_path, canvas)



def process_images(data, out_dir, num_samples=100, seed=1234):
    random.seed(seed)
    data = data.shuffle(seed=seed).select(range(num_samples))
    
    for i, example in enumerate(data):
        # Load the image from a URL or file path
        # img = example['image']
        # # print(img)
        # output_path = os.path.join(out_dir, "gray_canny_inv_filled_distorted_handwrite", f'{i}.png')
        # convert_to_sketch(img, output_path)

        # original_path = os.path.join(out_dir, "original", f'{i}.png')
        # img.save(original_path)
        
        html = example['text']
        code_path = os.path.join(out_dir, "original", f'{i}.html')
        with open(code_path, 'w') as f:
            f.write(html)

print("hello world!!")
# dataset = load_dataset("/juice2/scr2/nlp/pix2code/zyanzhe/WebSight_82k_train/", split='train[:100]')
dataset = load_from_disk("/juice2/scr2/nlp/pix2code/zyanzhe/WebSight_82k_train/")
print(dataset)
out_dir = "/sailhome/lansong/Sketch2Code/cv2_sketch"

process_images(dataset, out_dir)