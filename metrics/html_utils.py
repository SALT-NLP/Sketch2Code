import os
import io
import json
import traceback
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright
from tqdm import tqdm

from shapely.geometry import box
from shapely.ops import unary_union


def boxes_adjacent(box1, box2, align_tolerance=8, adj_tolerance=4):
    """
    Determine if two boxes are adjacent considering a horizontal and vertical tolerance.
    Boxes are considered adjacent if:
    - They are nearly aligned vertically (their vertical centers are close within y_tolerance)
    - They are horizontally sequential without a large gap (distance between them does not exceed x_tolerance)
    - OR they are vertically sequential where one box is directly above or below the other within y_tolerance
    """
    # Calculate vertical centers and horizontal centers
    vertical_center1 = box1['y'] + box1['height'] / 2
    vertical_center2 = box2['y'] + box2['height'] / 2

    horizontal_center1 = box1['x'] + box1['width'] / 2
    horizontal_center2 = box2['x'] + box2['width'] / 2
    
    # Check vertical alignment
    vertically_aligned = abs(vertical_center1 - vertical_center2) <= align_tolerance
    
    # Check horizontal adjacency
    horizontally_adjacent = (box1['x'] + box1['width'] + adj_tolerance >= box2['x'] and box1['x'] < box2['x']) or \
                            (box2['x'] + box2['width'] + adj_tolerance >= box1['x'] and box2['x'] < box1['x'])
    
    # Check horizontal alignment for vertical adjacency
    horizontally_aligned = abs(horizontal_center1 - horizontal_center2) <= align_tolerance

    # Check vertical adjacency
    vertically_adjacent = (box1['y'] + box1['height'] + adj_tolerance >= box2['y'] and box1['y'] < box2['y']) or \
                          (box2['y'] + box2['height'] + adj_tolerance >= box1['y'] and box2['y'] < box1['y'])
    
    return (vertically_aligned and horizontally_adjacent) or (horizontally_aligned and vertically_adjacent)


def merge_boxes(box1, box2):
    """ Merge two adjacent boxes into a larger box """
    x1 = min(box1['x'], box2['x'])
    y1 = min(box1['y'], box2['y'])
    x2 = max(box1['x'] + box1['width'], box2['x'] + box2['width'])
    y2 = max(box1['y'] + box1['height'], box2['y'] + box2['height'])
    return {
        'x': x1,
        'y': y1,
        'width': x2 - x1,
        'height': y2 - y1
    }

def is_within(box1, box2):
    """ Check if box1 is within box2 """
    return (box1['x'] >= box2['x'] and 
            box1['y'] >= box2['y'] and 
            box1['x'] + box1['width'] <= box2['x'] + box2['width'] and 
            box1['y'] + box1['height'] <= box2['y'] + box2['height'])
    

def extract_visual_components(url, save_path):
    # Convert local path to file:// URL if it's a file
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)

    screenshot_image = None
    element_data = {}
    
    try:
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch()
            page = browser.new_page()

            # Navigate to the URL
            page.goto(url, timeout=60000)
            
            total_width = page.evaluate("() => document.documentElement.scrollWidth")
            total_height = page.evaluate("() => document.documentElement.scrollHeight")

            # Define selectors for specific types of elements
            # selectors = {
            #     'image': 'img',
            #     'text_block': 'p, span, h1, h2, h3, h4, h5, h6, div:not(:has(span, p, h1, h2, h3, h4, h5, h6, a))',
            #     'form_table': 'form, table',
            #     'button': 'button, input[type="button"], input[type="submit"]',
            #     'nav_bar': 'nav',
            #     'divider': 'hr, div[class*="separator"]'  # Adjust class selector based on actual usage
            # }
            selectors = {
                'video': 'video',
                'image': 'img',
                'text_block': 'p, span, a, strong, h1, h2, h3, h4, h5, h6, li, th, td, label, code, pre, div',
                'form_table': 'form, table, div.form',
                'button': 'button, input[type="button"], input[type="submit"], [role="button"]',
                'nav_bar': 'nav, [role="navigation"], .navbar, [class~="nav"], [class~="navigation"], [class~="menu"], [class~="navbar"], [id="menu"], [id="nav"], [id="navigation"], [id="navbar"]',
                'divider': 'hr, [class*="separator"], [class*="divider"], [id="separator"], [id="divider"], [role="separator"]',
            }
            
            for key, value in selectors.items():
                element_data[key] = []
                elements = page.query_selector_all(value)
                for element in elements:
                    if not element.is_visible():
                        continue
                    
                    box = element.bounding_box()
                    if box and box['width'] > 0 and box['height'] > 0:  # Only add if the element has a visible bounding box
                        text_content = None
                        if key == 'text_block':
                            is_direct_text = element.evaluate("""
                                (el) => Array.from(el.childNodes).some(node => 
                                    node.nodeType === Node.TEXT_NODE && node.textContent.trim() !== '')
                            """)
                            tag_name = element.evaluate("el => el.tagName.toLowerCase()")
                            if tag_name == 'div' and not is_direct_text:  # Skip div elements that do not contain direct text
                                continue
                            
                            text_content = element.text_content().strip()
                            if not text_content:  # Skip empty text blocks
                                continue
                            
                        element_data[key].append({
                            'type': key,
                            'box': {'x': box['x'], 'y': box['y'], 'width': box['width'], 'height': box['height']},
                            # 'size': {'width': box['width'], 'height': box['height']},
                            'text_content': text_content,
                        })

            # Process adjacent text blocks
            text_blocks = element_data['text_block']
            text_blocks.sort(key=lambda block: (block['box']['y'], block['box']['x']))
            merged_text_blocks = []
            while text_blocks:
                current = text_blocks.pop(0)
                index = 0
                add = True
                while index < len(text_blocks):
                    if is_within(text_blocks[index]['box'], current['box']):
                        # Skip nested text blocks
                        del text_blocks[index]
                        continue
                    elif is_within(current['box'], text_blocks[index]['box']):
                        # the current box is nested within another box, skip the current box
                        add = False
                        break
                    
                    if boxes_adjacent(current['box'], text_blocks[index]['box']):
                        # print("MERGING:")
                        # print(current)
                        # print()
                        # print(text_blocks[index])
                        # print()
                        if current['box']['x'] < text_blocks[index]['box']['x'] or current['box']['y'] < text_blocks[index]['box']['y']:
                            current['text_content'] += " " + text_blocks[index]['text_content']
                        else:
                            current['text_content'] = text_blocks[index]['text_content'] + " " + current['text_content']
                        
                        current['box'] = merge_boxes(current['box'], text_blocks[index]['box'])
                        # print("merged box:")
                        # print(current['box'])
                        # print()
                        del text_blocks[index]
                    else:
                        index += 1
                
                if add:
                    merged_text_blocks.append(current)

            # Combine merged text blocks with other elements
            element_data['text_block'] = merged_text_blocks

            # print(json.dumps(element_data, indent=4))
            
            # Take the full page screenshot
            image_bytes = page.screenshot(full_page=True, animations="disabled", timeout=60000)

            # Convert bytes to a PIL Image
            image_buffer = io.BytesIO(image_bytes)
            screenshot_image = Image.open(image_buffer)

            # Draw bounding boxes and labels
            draw = ImageDraw.Draw(screenshot_image)
            # font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            font = ImageFont.load_default()  # Using default font

            for key in element_data:
                for item in element_data[key]:
                    x = item['box']['x']
                    y = item['box']['y']
                    width = item['box']['width']
                    height = item['box']['height']
                    draw.rectangle(((x, y), (x + width, y + height)), outline="red", width=2)
                    draw.text((x, y), item['type'], fill="red", font=font)
                # if item['type'] == 'text_block':
                #     draw.text((x, y), item['text'], fill="red", font=font)
                # else:
                #     draw.text((x, y), item['type'], fill="red", font=font)

            # Save the annotated image to a file
            if save_path:
                screenshot_image.save(save_path)
            
            
            # finally, normalize the positions and sizes of each element to relative values before returning
            for key in element_data:
                for item in element_data[key]:
                    box = item['box']
                    item['box'] = {
                        'x': box['x'] / total_width,
                        'y': box['y'] / total_height,
                        'width': box['width'] / total_width,
                        'height': box['height'] / total_height
                    }

            browser.close()
    except Exception as e:
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        print(traceback.format_exc())
        screenshot_image = Image.new('RGB', (1280, 960), color='white')
        if save_path:
            screenshot_image.save(save_path)

    return element_data


def intersection(boxA, boxB):
    xA = max(boxA['x'], boxB['x'])
    yA = max(boxA['y'], boxB['y'])
    xB = min(boxA['x'] + boxA['width'], boxB['x'] + boxB['width'])
    yB = min(boxA['y'] + boxA['height'], boxB['y'] + boxB['height'])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    
    return inter_width * inter_height


def compute_list_iou(listA, listB):
    total_intersection = 0
    total_area_A = 0
    total_area_B = 0

    for elemA in listA:
        boxA = elemA['box']
        total_area_A += boxA['width'] * boxA['height']
        for elemB in listB:
            boxB = elemB['box']
            total_intersection += intersection(boxA, boxB)

    for elemB in listB:
        boxB = elemB['box']
        total_area_B += boxB['width'] * boxB['height']

    total_union = total_area_A + total_area_B - total_intersection

    # Calculate the Intersection over Union (IoU)
    total_iou = total_intersection / total_union if total_union > 0 else 0

    return total_iou, total_area_A + total_area_B

def compute_weighted_iou(elementsA, elementsB):
    areas = {}
    ious = {}
    
    all_keys = set(elementsA.keys()).union(set(elementsB.keys()))
    
    for key in all_keys:       # elementsB is the reference layout
        if key not in elementsA:
            elementsA[key] = []
        if key not in elementsB:
            elementsB[key] = []
        
        ious[key], areas[key] = compute_list_iou(elementsA[key], elementsB[key])
    
    total_area = sum(areas[key] for key in all_keys)
    weighted_iou = sum(areas[key] * ious[key] for key in all_keys) / total_area if total_area > 0 else 0
    
    return weighted_iou

def bounding_box_to_polygon(bbox):
    return box(bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height'])

def compute_list_iou_shapely(listA, listB):
    if not listA and not listB:
        # If both lists are empty, IoU is 0 and union area is 0
        return 0.0, 0.0
    elif not listA:
        # If listA is empty, IoU is 0 and union area is the area of listB
        polygonsB = [bounding_box_to_polygon(elem['box']) for elem in listB]
        unionB = unary_union(polygonsB)
        union_area = unionB.area
        return 0.0, union_area
    elif not listB:
        # If listB is empty, IoU is 0 and union area is the area of listA
        polygonsA = [bounding_box_to_polygon(elem['box']) for elem in listA]
        unionA = unary_union(polygonsA)
        union_area = unionA.area
        return 0.0, union_area
    
    # Convert bounding boxes to shapely polygons
    polygonsA = [bounding_box_to_polygon(elem['box']) for elem in listA]
    polygonsB = [bounding_box_to_polygon(elem['box']) for elem in listB]

    # Create a union of all polygons in listA and listB
    unionA = unary_union(polygonsA)
    unionB = unary_union(polygonsB)

    # Compute the total intersection and union areas
    intersection_area = unionA.intersection(unionB).area
    union_area = unionA.union(unionB).area

    total_iou = intersection_area / union_area if union_area > 0 else 0

    return total_iou, union_area

def compute_weighted_iou_shapely(elementsA, elementsB):
    areas = {}
    ious = {}
    
    all_keys = set(elementsA.keys()).union(set(elementsB.keys()))
    
    for key in all_keys:       # elementsB is the reference layout
        if key not in elementsA:
            elementsA[key] = []
        if key not in elementsB:
            elementsB[key] = []

        ious[key], areas[key] = compute_list_iou_shapely(elementsA[key], elementsB[key])
    
    total_area = sum(areas[key] for key in all_keys)
    weighted_iou = sum(areas[key] * ious[key] for key in all_keys) / total_area if total_area > 0 else 0
    
    multi_score = {}
    for key in all_keys:
        multi_score[key] = (ious[key], areas[key] / total_area)     # score, weight
    
    return weighted_iou, multi_score


if __name__ == "__main__":
    # filename = "annotated_screenshot.png"
    # save_path = os.path.join(os.getcwd(), filename)
    # extract_visual_components("/sailhome/lansong/Sketch2Code/sketch2code_sample/247.html", save_path)
    
    input_dir = '/juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_dataset_v1'
    out_dir = '/sailhome/lansong/Sketch2Code/sample_layout_extraction'
    all_files = os.listdir(input_dir)
    
    examples = []
    for filename in all_files:
        # if '-' in filename and filename.endswith('.png'):
        if '_' not in filename and filename.endswith('.png'):
            examples.append(filename)
    
    examples = examples[:20]
    
    for filename in tqdm(examples):
        img_id = filename.split(".")[0]
        html_path = os.path.join(input_dir, f'{img_id}.html')
        save_path = os.path.join(out_dir, f'{img_id}.png')
        # print(f"processing file {img_id}")
        extract_visual_components(html_path, save_path)