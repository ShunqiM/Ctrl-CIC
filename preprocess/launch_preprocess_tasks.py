import os, sys
sys.path.insert(0, os.path.abspath("."))

from api_mmwebpage.ImageDataParser import ImageDataParser
from api_mmwebpage.TextDataParser import TextDataParser
from starter.env_getter import get_env
from starter.env_setter import set_visible_gpus

SPLITS = get_env('SPLITS')

if __name__ == '__main__':
    # Usage: "python -m preprocess/launch_download_tasks --split train --device 2" or "python -m preprocess/launch_download_tasks.py"

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='download_img', choices={'download_img', 'extract_txt'})
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--device', type=str, default=None) # different main-GPU is required when processing train/val/test in parallel
    args = parser.parse_args()
    
    if args.task == 'extract_txt':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        data_parser = TextDataParser()
        if args.split is not None:
            assert args.split in SPLITS, args.split
            data_parser.extract_txt(split=args.split)
        else:
            for split in SPLITS:
                data_parser.extract_txt(split=split)
        exit()
        
    if args.device: set_visible_gpus(args.device)

    data_parser = ImageDataParser()

    if args.split is not None:
        assert args.split in SPLITS, args.split
        data_parser.preprocess_images(split=args.split, task=args.task)
    else:
        for split in SPLITS:
            data_parser.preprocess_images(split=split, task=args.task)