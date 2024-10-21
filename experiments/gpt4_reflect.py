import os
import traceback
import json
import argparse
import shutil
import time

from openai import OpenAI, AzureOpenAI
from tqdm import tqdm
from metrics.visual_score import visual_eval_v3_multi
from metrics.layout_similarity import layout_similarity
from utils.prompt_utils import encode_image, get_reflection_message, get_direct_system_message, get_direct_text_augmented_user_message, get_user_message_with_qa_pairs
from utils.utils import extract_html, remove_html_comments, read_html, extract_text_from_html, extract_feedback, rescale_image_loader, gpt_cost
from utils.screenshot import take_and_save_screenshot

"""
nlprun -m sphinx1 -g 0 -c 4 -r 40G -a sketch2code --output /sailhome/lansong/Sketch2Code/logs/gpt4o_mini_reflect_v0.2.txt 'python gpt4_reflect.py --model gpt-4o-mini'
"""

total_input_tokens = 0
total_output_tokens = 0
total_cost = 0

def generate(model, openai_client, screenshot_path, sketch_path, source_html_path, out_dir, img_id):
    global total_input_tokens, total_output_tokens, total_cost
    direct_system_message = get_direct_system_message()
    
    html = remove_html_comments(read_html(source_html_path))
    html_text = extract_text_from_html(html)
    sketch = rescale_image_loader(sketch_path)
    screenshot = rescale_image_loader(screenshot_path)
    direct_user_message = get_direct_text_augmented_user_message(sketch, html_text)


    html_response = None
    turn = 0
    max_turns = 5
    qa_pairs = ""
    num_questions = 0
    res_dict = {"id": img_id, "results": []}
    
    agent_messages = [
        {
            "role": "system",
            "content": direct_system_message,
        },
        {
            "role": "user",
            "content": direct_user_message,
        }
    ]
    
    for _ in range(3):
        try:
            curr_image = None
            
            # hack: reuse the existing directing prompting output if possible
            # text_augmented_dir = "/juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_eval/v0.2/gpt4o_text_augmented" if model == "gpt-4o" else "/juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_eval/v0.2/gpt4v_text_augmented"
            # if os.path.isfile(os.path.join(text_augmented_dir, f"{img_id}.html")) and os.path.isfile(os.path.join(text_augmented_dir, f"{img_id}.png")):
            #     curr_image = rescale_image_loader(os.path.join(text_augmented_dir, f"{img_id}.png"))
            #     direct_html = read_html(os.path.join(text_augmented_dir, f"{img_id}.html"))
            # else:

            # first get the direct prompting result
            response = openai_client.chat.completions.create(
                model=model,
                messages=agent_messages,
                max_tokens=4096,
                temperature=0.0,
            )
            prompt_tokens, completion_tokens, cost = gpt_cost(model, response.usage)
            total_input_tokens += prompt_tokens
            total_output_tokens += completion_tokens
            total_cost += cost
            
            agent_resp = response.choices[0].message.content.strip()
            print(f"agent: {agent_resp}")
            
            direct_html = extract_html(agent_resp)
            
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
            
            if not curr_image:
                img_path = os.path.join(out_dir, f'{img_id}_0.png')
                take_and_save_screenshot(html_path, output_file=img_path, do_it_again=True)
                curr_image = rescale_image_loader(img_path)
            
            feedback = None
            while turn < max_turns:
                turn += 1
                
                if not feedback:
                # first, get reflection feedback from the simulated user
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=get_reflection_message(screenshot, curr_image),
                        max_tokens=4096,
                    )
                    prompt_tokens, completion_tokens, cost = gpt_cost(model, response.usage)
                    total_input_tokens += prompt_tokens
                    total_output_tokens += completion_tokens
                    total_cost += cost
                    
                    user_resp = response.choices[0].message.content.strip()
                    
                    print(f"simulated user: {user_resp}")
                    
                    feedback = extract_feedback(user_resp)
                
                if not feedback or "generation complete" in feedback.lower():
                    break
                
                # the simulated user has provided feedback, improve the implementation based on the feedback
                # continue with the generation
                agent_messages.append({
                    "role": "assistant",
                    "content": agent_resp,
                })
                
                agent_messages.append({
                    "role": "user",
                    "content": feedback,
                })
                
                
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=agent_messages,
                    max_tokens=4096,
                    temperature=0.0,
                )
                prompt_tokens, completion_tokens, cost = gpt_cost(model, response.usage)
                total_input_tokens += prompt_tokens
                total_output_tokens += completion_tokens
                total_cost += cost
                
                agent_resp = response.choices[0].message.content.strip()
                print(f"agent: {agent_resp}")
                
                html_output = extract_html(agent_resp)
                if not html_output:
                    continue
                
                html_path = os.path.join(out_dir, f'{img_id}_{turn}.html')
                with open(html_path, 'w') as f:
                    f.write(html_output)
                    
                img_path = os.path.join(out_dir, f'{img_id}_{turn}.png')
                take_and_save_screenshot(html_path, output_file=img_path, do_it_again=True)
                curr_image = rescale_image_loader(img_path)
                

                res_dict["results"].append({
                    "filename": html_path,
                    "feedback": feedback,
                })
                feedback = None
                
                
            
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
    
    return False, 0, {}
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--input_dir', type=str, default='/juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_dataset_v1')
    parser.add_argument('--out_dir', type=str, default='/juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_eval/v0.2/gpt4o_mini_reflect')
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
    num_img_processed = 0
    
    for img_id in tqdm(examples, total=args.ends_at):
        num_img_processed += 1
        if total > args.ends_at:
            print(f"ending early after processing {args.ends_at} webpages")
            break
        if num_img_processed <= args.starts_from:
            continue
        screenshot_path = os.path.join(input_dir, f'{img_id}.png')
        html_path = os.path.join(input_dir, f'{img_id}.html')
        for sketch_file in examples[img_id]:
            sketch_id = sketch_file.split('.')[0]
            total += 1
            sketch_path = os.path.join(input_dir, sketch_file)
            success, num_turns, res_dict = generate(model, openai_client, screenshot_path, sketch_path, html_path, out_dir, sketch_id)
            total_turns += num_turns
            
            if success:
                num_success += 1
                res_dicts.append(res_dict)
                print(f"successfully generated webpage for {sketch_id} after {num_turns} turns")
                if num_turns > 1:
                    num_asked += 1
        
        # num_img_processed += 1
        
        if num_img_processed % 10 == 0:
            with open(os.path.join(out_dir, 'res_dict.json'), "w") as f:
                json.dump(res_dicts, f, indent=4)
    
    with open(os.path.join(out_dir, 'res_dict.json'), "w") as f:
        json.dump(res_dicts, f, indent=4)
        
    print(f"All done. {num_success} out of {total} successful generations.")
    print(f"The agent asked clarifying question(s) in {num_asked} out of {num_success} generations.")
    print(f"The average number of turns taken for each generation is {total_turns / num_success}")
    
    
    print(f"Total input prompt tokens: {total_input_tokens}")
    print(f"Total output prompt tokens: {total_output_tokens}")
    print(f"Total cost: {total_cost}")