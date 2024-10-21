import os
import traceback
import json
import argparse
import shutil
import time

from openai import OpenAI, AzureOpenAI
from tqdm import tqdm

from utils.prompt_utils import get_direct_system_message, get_direct_user_message
from utils.utils import extract_html, extract_all_questions, remove_html_comments, extract_title, read_html, gpt_cost
from metrics.layout_similarity import layout_similarity
from utils.screenshot import take_and_save_screenshot


"""
nlprun -m sphinx2 -g 0 -c 4 -r 40G -a sketch2code --output /sailhome/lansong/Sketch2Code/logs/gpt4o_mini_direct_v0.2.txt 'python /sailhome/lansong/Sketch2Code/Sketch2Code/gpt4_direct.py --model gpt-4o-mini --out_dir /juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_eval/v0.2/gpt4o_mini_direct'
"""


total_input_tokens = 0
total_output_tokens = 0
total_cost = 0


def generate(model, openai_client, screenshot_path, sketch_path, source_html_path, out_dir, img_id):
    global total_input_tokens, total_output_tokens, total_cost
    
    html = remove_html_comments(read_html(source_html_path))
    topic = extract_title(html)
    
    agent_system_message = get_direct_system_message()
    agent_user_message = get_direct_user_message(sketch_path, topic)
    # print(agent_system_message)
    # print(agent_user_message)

    agent_messages = [
        {
            "role": "system",
            "content": agent_system_message
        },
        {
            "role": "user",
            "content": agent_user_message,
        }
    ]


    html_response = None
    i = 0
    max_tries = 10

    while not html_response and i < max_tries:
        try:
            i += 1
            response = openai_client.chat.completions.create(
                model=model,
                messages=agent_messages,
                max_tokens=2048,
            )
            prompt_tokens, completion_tokens, cost = gpt_cost(model, response.usage)
            total_input_tokens += prompt_tokens
            total_output_tokens += completion_tokens
            total_cost += cost
            output_texts = response.choices[0].message.content.strip()
            
            print(f"agent: {output_texts}")
            
            html_response = extract_html(output_texts)
            if not html_response:
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
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--input_dir', type=str, default='/juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_dataset_v1')
    parser.add_argument('--out_dir', type=str, default='/juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_eval/v0.2/gpt4o_direct')
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
    
    openai_client = OpenAI()

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
        
        screenshot_path = os.path.join(input_dir, f'{img_id}.png')
        html_path = os.path.join(input_dir, f'{img_id}.html')

        for sketch_file in examples[img_id]:
            sketch_id = sketch_file.split('.')[0]
            total += 1
            sketch_path = os.path.join(input_dir, sketch_file)
            success, res_dict = generate(model, openai_client, screenshot_path, sketch_path, html_path, out_dir, sketch_id)
            
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
    print(f"Total input prompt tokens: {total_input_tokens}")
    print(f"Total output prompt tokens: {total_output_tokens}")
    print(f"Total cost: {total_cost}")