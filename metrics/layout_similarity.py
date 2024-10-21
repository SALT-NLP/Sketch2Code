import os
import shutil
import json

from tqdm import tqdm
from metrics.html_utils import extract_visual_components, compute_weighted_iou, compute_weighted_iou_shapely


def layout_similarity(input_list, debug=False):
    predict_html_list, original_html = input_list[0], input_list[1]
    results = []
    multi_scores = []
    
    reference_save_path = None
    if debug:
        reference_save_path = original_html.replace(".html", "_p.png")
    reference_layout = extract_visual_components(original_html, reference_save_path)
    
    for predict_html in predict_html_list:
        
        predict_save_path = predict_html.replace(".html", "_p.png") if debug else None
        predict_layout = extract_visual_components(predict_html, predict_save_path)
        
        iou_score, multi_score = compute_weighted_iou_shapely(predict_layout, reference_layout)
        results.append(iou_score)
        multi_scores.append(multi_score)
    
    return results, multi_scores


if __name__ == "__main__":    
    input_dir = '/juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_dataset_v1'
    out_dir = '/sailhome/lansong/Sketch2Code/sketch2code_sample_eval/layout_scores'
    
    os.makedirs(out_dir, exist_ok=True)
    
    with open("/juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_eval/v0/gpt4o_text_augmented/res_dict.json", "r") as f:
        res_dict = json.load(f)
    res_dict = res_dict[:20]
    
    
    results = []
    for item in tqdm(res_dict):
        if len(item["results"]) <= 1:
            continue
        
        first = item["results"][0]
        final = item["results"][-1]
        
        sketch_id = item["id"]
        img_id = sketch_id.split('_')[0]
        
        first_file = os.path.join(out_dir, f"{sketch_id}_0.html")
        first_img = os.path.join(out_dir, f"{sketch_id}_0.png")
        final_file = os.path.join(out_dir, f"{sketch_id}_1.html")
        final_img = os.path.join(out_dir, f"{sketch_id}_1.png")
        
        source_file = os.path.join(out_dir, f"{img_id}.html")
        source_img = os.path.join(out_dir, f"{img_id}.png")
        
        
        # copy all of them to out_dir
        shutil.copy(first['filename'], first_file)
        shutil.copy(first['filename'].replace(".html", ".png"), first_img)
        
        shutil.copy(final['filename'], final_file)
        shutil.copy(final['filename'].replace(".html", ".png"), final_img)
        
        shutil.copy(os.path.join(input_dir, f"{img_id}.html"), source_file)
        shutil.copy(os.path.join(input_dir, f"{img_id}.png"), source_img)
        
        
        pred_list = [first_file, final_file, source_file]       # sanity check: IoU of source file against itself should be 1
        input_list = [pred_list, source_file]
        
        scores, _ = layout_similarity(input_list, debug=True)
        assert len(scores) == 3
        
        results.append({
            "first_file": first_file,
            "first_score": scores[0],
            "second_file": final_file,
            "second_score": scores[1],
            "source_file": source_file,
            "source_score": scores[2],
        })
    
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)