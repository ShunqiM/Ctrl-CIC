# Ctrl-CIC: Directing the Visual Narrative through User-Defined Highlights

Repository for ["Controllable Contextualized Image Captioning: Directing the Visual Narrative through User-Defined Highlights"](https://ctrl-cic.github.io/) in ECCV 2024.

## Setup

- Download the WikiWeb2M dataset from [WikiWeb2M](https://github.com/google-research-datasets/wit/blob/main/wikiweb2m.md).
- Update your local path variables in [env_definer.sh](starter/env_definer.sh) and setup them according to the [ReadME](starter/ReadMe.md) file.
- Run preprocess/launch_preprocess_tasks.py with task flag 'download_img' and 'extract_txt' respectively, to download the images required for the dataset (around 3~4 TB) and extract the text data locally.
- Some preprocessed data are provided, including
    - .csv files specifying the training, validation and testing splits
    - Extracted [highlights](data/highlights/) for inference during evaluation.
    - [GRIT image captions](data/images_1000_response.json) facilitating text-based GPT-4 Ctrl-CIC caption generation.

    Remeber to move these files to the corresponding local path you specified. 

- Extract CLIP image feature with `scripts/extract_image_features.py` for efficient local training and evaluation. 
- Generate relevance scores for pseudo training highlights with `scripts/mask_generation.py`


## Finetune

- Run the training program with the corresponding config file, for example, <br>`python cli/train.py --config experiments/finetune/longt5.yaml`

## Inference

- For traditional CIC tasks, refer to [eval_configs](experiments/eval_configs/). Update the run_id according to your local checkpoints and run the inference scripts. <br> `python cli/eval.py --config experiments/eval_configs/eval_full.yaml`. CIC performance will be recorded during inference.
- For Ctrl-CIC tasks, first generate Ctrl-CIC captions with [ccic_eval_configs](experiments/ccic_eval_configs/).


## Evaluation
The Ctrl-CIC captions can be evaluated as follows:
- CLIPScore and CLIPScore-Sentence by setting use_clip_score and use_sent_score and load_predictions to True, and run the evaluation scripts again.
- Recall, with `python scripts/calculate_recall.py`
- Diversity, with `python scripts/diversity_eval.py`
- GPT-4(V) empowered metrics, by 
    - First generate jsons files to be uploaded with `python scripts/generate_prompt.py --task eval`
    - Update your openai key [here](utils/tools.py)
    - Run `python scripts/query_response.py` for GPT-4(V) API call
    - Run `python scripts/get_gpt_scores.py` to compute the GPT-4(V) evaluation metrics scores.

## Pretrained Weights
The pretrained weights are avaliable at [huggingface](https://huggingface.co/Shunqi/Ctrl-CIC).

## Demo
For interactive Ctrl-CIC demo, you can run `python scripts/rctrl_inference.py` which allows flexible selection of the highlights and image. A similar program is provided for p-ctrl, but the output is shown on the command line.

## Acknowledgement
The dataset and data loading implementation is based on the code provided in [WikiWeb2M](https://github.com/google-research-datasets/wit/blob/main/wikiweb2m.md).

## Citation  
```
@InProceedings{Mao_2024_ECCV,
    author    = {Mao, Shunqi and Zhang, Chaoyi and Su, Hang and Song, Hwanjun and Shalyminov, Igor and Cai, Weidong},
    title     = {Controllable Contextualized Image Captioning: Directing the Visual Narrative through User-Defined Highlights},
    booktitle = {Proceedings of the 18th European Conference on Computer Vision (ECCV)},
    year      = {2024}
}
```