import os
import traceback
import json
import argparse
import shutil
import torch
import time

import retry
import mimetypes
import base64
from PIL import Image
import PIL
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from metrics.visual_score import visual_eval_v3_multi
from metrics.layout_similarity import layout_similarity
from utils.prompt_utils import get_reflection_message, get_direct_system_message, get_direct_user_prompt, get_user_message_with_qa_pairs
from utils.utils import extract_html, remove_html_comments, read_html, extract_text_from_html, extract_feedback, gpt_cost, rescale_image_loader, gemini_encode_image, cleanup_response, extract_title
from utils.screenshot import take_and_save_screenshot

"""
nlprun -m sphinx8 -g 1 -c 4 -r 40G -a sketch2code --output /sailhome/lansong/Sketch2Code/logs/llava_direct_v0.2.txt 'python llava_direct.py'
"""


def llava_call(model, processor, user_message, image, history=None):
    def parse_resp(text_output):
        idx = text_output.rfind("assistant")

        if idx > -1:
            return text_output[idx+len("assistant"):].strip()
        else:
            return text_output
    
    if not history:
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {"type": "image"},
                ],
            },
        ]
    else:
        conversation = history
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
            ],
        })
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    output = parse_resp(processor.decode(model.generate(**inputs, max_new_tokens=4096, temperature=0.5, repetition_penalty=1.1)[0], skip_special_tokens=True))
    
    conversation.append({
        "role": "assistant",
        "content": [
            {"type": "text", "text": output}
        ]
    })

    return output, conversation

def generate(model, processor, screenshot_path, sketch_path, source_html_path, out_dir, img_id):    
    
    direct_system_message = get_direct_system_message()
    
    html = remove_html_comments(read_html(source_html_path))
    topic = extract_title(html)
    sketch = gemini_encode_image(sketch_path)
    screenshot = rescale_image_loader(screenshot_path)
    direct_user_prompt = get_direct_user_prompt(topic)

    html_response = None

    res_dict = {"id": img_id, "results": []}

    for _ in range(3):
        try:
            history = None
            turn = 0

            # first get the direct prompting result
            agent_resp, history = llava_call(model, processor, direct_system_message + '\n\n' + direct_user_prompt, sketch, history)
            print(f"agent: {agent_resp}")
            
            direct_html = cleanup_response(extract_html(agent_resp))
            
            if not direct_html:
                print(f"Failed to generate direct html for image {img_id}")
                direct_html = ""
            
            html_path = os.path.join(out_dir, f'{img_id}_0.html')
            with open(html_path, 'w') as f:
                f.write(direct_html)
            
            res_dict["results"].append({
                "filename": html_path,
                "feedback": "N/A",  # feedback received in the previous round
            })
                
            
            pred_list = []
            for i in range(len(res_dict["results"])):
                pred_list.append(res_dict["results"][i]["filename"])
            
            input_list = [pred_list, source_html_path]
            return_score_list = visual_eval_v3_multi(input_list)
            prev_score = 0
            
            final_layout_scores, layout_multi_scores = layout_similarity(input_list)
            
            for i in range(len(res_dict["results"])):
                _, final_score, multi_score = return_score_list[i]
                final_size_score, final_matched_text_score, final_position_score, final_text_color_score, final_clip_score = multi_score
                
                res_dict["results"][i]["scores"] = {
                    "Final Score": final_score,
                    "Block-Match": final_size_score,
                    "Text": final_matched_text_score,
                    "Position": final_position_score,
                    "Color": final_text_color_score,
                    "CLIP": final_clip_score,
                }
                res_dict["results"][i]["layout_scores"] = {
                    "final_layout_score": final_layout_scores[i],
                    "layout_multi_score": layout_multi_scores[i],
                }
                res_dict["results"][i]["delta_score"] = final_score - prev_score
                prev_score = final_score
            
            return True, turn, res_dict
        except Exception as e: 
            print(f"Webpage generation for {img_id} failed dur to {e}, retrying...")
            print(traceback.format_exc())
            time.sleep(3)
    
    return False, 0, res_dict
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llava-hf/llama3-llava-next-8b-hf')
    parser.add_argument('--input_dir', type=str, default='/juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_dataset_v1')
    parser.add_argument('--out_dir', type=str, default='/juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_eval/v0.2/llava_direct')
    parser.add_argument('--starts_from', type=int, default=0)
    parser.add_argument('--ends_at', type=int, default=50)
    parser.add_argument("--sanity_check", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    input_dir = args.input_dir
    out_dir = args.out_dir
    model_name = args.model
    
    os.makedirs(out_dir, exist_ok=True)
    
    # copy "rick.jpg" to the target directory
    source_image_path = "experiments/rick.jpg"
    destination_image_path = os.path.join(out_dir, "rick.jpg")
    shutil.copy(source_image_path, destination_image_path)
    
    processor = LlavaNextProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name, 
        device_map="auto", 
        load_in_8bit=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16
    )

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
    total = 0
    res_dicts = []
    num_img_processed = 0
    
    for img_id in tqdm(examples, total = args.ends_at):
        num_img_processed += 1
        if total > args.ends_at:
            print(f"ending early after processing {args.ends_at} webpages")
            break
        screenshot_path = os.path.join(input_dir, f'{img_id}.png')
        html_path = os.path.join(input_dir, f'{img_id}.html')
        for sketch_file in examples[img_id]:
            sketch_id = sketch_file.split('.')[0]
            total += 1
            if num_img_processed <= args.starts_from:
                continue
            sketch_path = os.path.join(input_dir, sketch_file)
            success, num_turns, res_dict = generate(model, processor, screenshot_path, sketch_path, html_path, out_dir, sketch_id)
            total_turns += num_turns
            res_dicts.append(res_dict)
            
            if success:
                num_success += 1
                print(f"successfully generated webpage for {sketch_id} after {num_turns} turns")
        
        # num_img_processed += 1
        
        if num_img_processed % 10 == 0:
            with open(os.path.join(out_dir, 'res_dict.json'), "w") as f:
                json.dump(res_dicts, f, indent=4)
    
    with open(os.path.join(out_dir, 'res_dict.json'), "w") as f:
        json.dump(res_dicts, f, indent=4)
        
    print(f"All done. {num_success} out of {total} successful generations.")
    print(f"The average number of turns taken for each generation is {total_turns / num_success}")