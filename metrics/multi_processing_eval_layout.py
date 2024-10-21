from metrics.visual_score import visual_eval_v3_multi
from metrics.layout_similarity import layout_similarity
from multiprocessing import Pool
import contextlib, joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import json
import os
import shutil

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


if __name__ == "__main__":
    debug = False
    multiprocessing = True

    orig_reference_dir = "/juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_dataset_v1"
    eval_name = "d2c_layout_test_gemini"

    ## copy the original reference directory to a new directory
    ## because we will be creating new screenshots
    reference_dir = "/juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_dataset_v1_" + eval_name
    os.makedirs(reference_dir, exist_ok=True)
    for filename in os.listdir(orig_reference_dir):
        if filename.endswith(".html") or filename == "rick.jpg":
            shutil.copy(os.path.join(orig_reference_dir, filename), os.path.join(reference_dir, filename))
    print ("copied original reference directory to ", reference_dir)

    # test_dirs = {
    #     "gpt4v_direct_prompting": "/sailhome/lansong/Sketch2Code/eval_results_pilot/gpt4v_direct",
    #     "gpt4v_conversation": "/sailhome/lansong/Sketch2Code/eval_results_pilot/gpt4v_conversation",
    #     "gpt4o_direct_prompting": "/sailhome/lansong/Sketch2Code/eval_results_pilot/gpt4o_direct",
    #     "gpt4o_conversation": "/sailhome/lansong/Sketch2Code/eval_results_pilot/gpt4o_conversation"
    # }
    test_dirs = {
        "gemini_direct_prompting": "/sailhome/lansong/gemini15flash_predictions_final/gemini_direct_prompting"
    }
    
    # file_name_list = set()
    file_name_list = []

    # check if the file is in all prediction directories
    for filename in os.listdir(reference_dir):
        if filename.endswith(".html"):
            if all([os.path.exists(os.path.join(test_dirs[key], filename)) for key in test_dirs]):
                file_name_list.append(filename)
    
    # for key in test_dirs:
    #     for filename in os.listdir(test_dirs[key]):
    #         if filename.endswith(".html"):
    #             title = filename.split('.')[0]
    #             parts = title.split('-')
    #             img_id = parts[0]
    #             # print(img_id)
    #             if not title.endswith('_p') and not title.endswith('_p_1') and os.path.exists(os.path.join(reference_dir, f"{img_id}.html")):
    #                 file_name_list.add(filename)

    # file_name_list = list(file_name_list)
    print ("total #egs: ", len(file_name_list))

    input_lists = []
    for filename in file_name_list:

        input_pred_list = [os.path.join(test_dirs[key], filename) for key in test_dirs]
        original = os.path.join(reference_dir, filename)
        # img_id = filename.split('-')[0]
        # original = os.path.join(reference_dir, f"{img_id}.html")

        input_list = [input_pred_list, original]
        input_lists.append(input_list)

    # print ("input_list: ", input_lists)
    if multiprocessing:
        with tqdm_joblib(tqdm(total=len(input_lists))) as progress_bar:
            return_score_lists = list(tqdm(Parallel(n_jobs=8)(delayed(layout_similarity)(input_list, debug=debug) for input_list in input_lists), total=len(input_lists)))
    else:
        return_score_lists = []
        for input_list in tqdm(input_lists):
            return_score_list = layout_similarity(input_list, debug=debug)
            return_score_lists.append(return_score_list)
        # print ("return lists: ", return_score_lists)
    
    res_dict = {}
    for key in test_dirs:
        res_dict[key] = []

    for i, filename in enumerate(file_name_list):
        idx = 0
        return_score_list = return_score_lists[i]
        # print ("return score list: ", return_score_list)
        if return_score_list:
            for key in test_dirs:
                final_layout_score, layout_multi_score = return_score_list[0][idx], return_score_list[1][idx]
                idx += 1
                res_dict[key].append({
                    "filename": filename,
                    "final_layout_score": final_layout_score,
                    "layout_multi_score": layout_multi_score,
                })
                # res_dict[key][filename] = final_layout_score
        else:
            print (filename + " didn't get a score")
            # for key in test_dirs:
            #     res_dict[key][filename] = 0

    ## cache all scores 
    with open("metrics/res_dict_{}.json".format(eval_name), "w") as f:
        json.dump(res_dict, f, indent=4)

    # for key in test_dirs:
    #     print(key)
    #     values = list(res_dict[key].values())
    #     # print (values)
    #     current_res = np.mean(np.array(values), axis=0)
    #     print(current_res)