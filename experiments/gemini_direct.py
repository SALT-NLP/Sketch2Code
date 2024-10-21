import os
import traceback
import json
import argparse
import shutil
import time

from tqdm import tqdm
import google.generativeai as genai
import retry
import mimetypes
import base64
from PIL import Image
import PIL

from utils.utils import extract_html, extract_all_questions, remove_html_comments, extract_title, read_html, cleanup_response, gemini_encode_image
from utils.prompt_utils import get_direct_prompt_combined
from metrics.layout_similarity import layout_similarity
from utils.screenshot import take_and_save_screenshot

"""
nlprun -m sphinx2 -g 0 -c 4 -r 40G -a sketch2code --output /sailhome/lansong/Sketch2Code/logs/gemini_mini_direct_v0.2.txt 'python /sailhome/lansong/Sketch2Code/Sketch2Code/gemini_direct.py --model gemini-1.5-flash --out_dir /juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_eval/v0.2/gemini_mini_direct'
"""


@retry.retry(tries=2, delay=2)
def gemini_call(gemini_client, encoded_image, prompt):
    generation_config = genai.GenerationConfig(
        temperature=0.,
        candidate_count=1,
        # max_output_tokens=4096,
    )

    safety_settings = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
    
    response = gemini_client.generate_content([prompt, encoded_image], generation_config=generation_config, safety_settings=safety_settings)
    response.resolve()

    if response.candidates[0].finish_reason != 1:
        print(response)

    response = response.text
    response = cleanup_response(response)

    return response


def generate(model, client, sketch_path, source_html_path, out_dir, img_id):
    html = remove_html_comments(read_html(source_html_path))
    topic = extract_title(html)
    sketch = gemini_encode_image(sketch_path)
    
    user_prompt = get_direct_prompt_combined(topic)
    # print(agent_system_message)
    # print(agent_user_message)


    html_response = None
    i = 0
    max_tries = 3

    while not html_response and i < max_tries:
        try:
            i += 1
            output_texts = gemini_call(client, sketch, user_prompt)
            
            print(f"agent: {output_texts}")
            
            html_response = cleanup_response(output_texts)
            if not html_response:
                print("warning: agent failed to generate valid html")
                continue

            print(f"final html: \n{html_response}")
            html_path = os.path.join(out_dir, f'{img_id}.html')
            with open(html_path, 'w') as f:
                f.write(html_response)
            
            img_path = os.path.join(out_dir, f'{img_id}.png')
            take_and_save_screenshot(html_path, output_file=img_path, do_it_again=True)
            
            pred_list = [html_path]
            
            input_list = [pred_list, source_html_path]
            final_layout_scores, layout_multi_scores = layout_similarity(input_list)
            
            assert len(final_layout_scores) == 1 and len(layout_multi_scores) == 1
            
            res_dict = {
                "id": img_id,
                "filename": html_path,
                "final_layout_score": final_layout_scores[0],
                "layout_multi_score": layout_multi_scores[0],
            }
            
            return True, res_dict
        except Exception as e: 
            print(f"Webpage generation for {img_id} failed dur to {e}")
            print(traceback.format_exc())
            time.sleep(3)
    return False, {}
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gemini-1.5-pro-latest')
    parser.add_argument('--input_dir', type=str, default='/juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_dataset_v1')
    parser.add_argument('--out_dir', type=str, default='/juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_eval/v0.2/genimi1.5pro_direct')
    parser.add_argument('--starts_from', type=int, default=0)
    parser.add_argument('--ends_at', type=int, default=50)
    parser.add_argument("--sanity_check", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    input_dir = args.input_dir
    out_dir = args.out_dir
    model = args.model
    
    os.makedirs(out_dir, exist_ok=True)
    
    # copy "rick.jpg" to the target directory
    source_image_path = "experiments/rick.jpg"
    destination_image_path = os.path.join(out_dir, "rick.jpg")
    shutil.copy(source_image_path, destination_image_path)
    
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    print(f"Running model {model}")
    client = genai.GenerativeModel(model)

    all_files = os.listdir(input_dir)
    if args.sanity_check:
        all_files = all_files[:5]
        print(all_files)
    
    examples = {}
    for filename in all_files:
        # if '-' in filename and filename.endswith('.png'):
        if '_' in filename and filename.endswith('.png'):
            # parts = filename.split('-')
            parts = filename.split('_')
            img_id = parts[0]
            
            if img_id not in examples:
                examples[img_id] = []
            
            examples[img_id].append(filename)
    
    num_success = 0
    total_turns = 0
    num_asked = 0
    total = 0
    res_dicts = []
    for img_id in tqdm(examples, total=args.ends_at):
        
        if total > args.ends_at:
            print(f"ending early after processing {args.ends_at} webpages")
            break
        
        html_path = os.path.join(input_dir, f'{img_id}.html')

        for sketch_file in examples[img_id]:
            sketch_id = sketch_file.split('.')[0]
            total += 1
            sketch_path = os.path.join(input_dir, sketch_file)
            success, res_dict = generate(model, client, sketch_path, html_path, out_dir, sketch_id)
            
            if success:
                num_success += 1
                res_dicts.append(res_dict)
                print(f"successfully generated webpage for {sketch_id}")
    
        if num_success % 10 == 0:
            with open(os.path.join(out_dir, 'res_dict.json'), "w") as f:
                json.dump(res_dicts, f, indent=4)
    
    with open(os.path.join(out_dir, 'res_dict.json'), "w") as f:
        json.dump(res_dicts, f, indent=4)
    
    print(f"All done. {num_success} out of {total} successful generations.")