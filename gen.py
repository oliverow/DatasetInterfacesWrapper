import argparse
import os
import sys
sys.path.append(os.path.join('dataset_interfaces'))
import torch
import matplotlib.pyplot as plt
from dataset_interfaces.dataset_interface import utils
from dataset_interfaces.dataset_interface import run_textual_inversion
from dataset_interfaces.dataset_interface import generate
from dataset_interfaces.dataset_interface import templates
import dataset_interfaces.dataset_interface.imagenet_utils as in_utils
import dataset_interfaces.dataset_interface.inference_utils as infer_utils

CLASS_TO_IDS = {
    'cat': [281, 282, 283, 284, 285, 286, 287], # cats x 7
    'dog': [153, 200, 229, 230, 235, 238, 239, 245, 248, 251, 252, 254, 256] # dogs x 13
}
ATTRIBUTES = [
    "green eye",
    "vertical pupil"
]
SHIFTS = [
    "in the snow",
    "in bright sunlight",
    "at night",
    "in a small town",
    "on a beach",
    "in a dense forest",
    "during a storm",
    "in a bustling city",
    "at sunrise",
    "at sunset",
    "in a desert landscape",
    "in a mountainous region",
    "under the rain",
    "in a foggy atmosphere",
    "in an urban park",
    "in a rural area",
]
PROMPT = "a photo of a <TOKEN> "

def gen_dataset(args, class_name):
    classes = CLASS_TO_IDS[class_name]
    class_names = [in_utils.IMAGENET_COMMON_CLASS_NAMES[c] for c in classes]

    root = args.encoder_root_path
    output_dir = os.path.join(args.output_dir, class_name)
    os.makedirs(output_dir, exist_ok=True)

    filename_pattern = "{}_{}.png"
    prompts = []

    prompt_template = PROMPT
    if args.with_edit:
        for attr in ATTRIBUTES:
            if args.with_shift:
                prompt_template += "with {} "
                prompts += list(map(lambda s: prompt_template.format(attr)+s, SHIFTS))
            else:
                prompts += list(map(lambda s: (s+" with {}").format('<TOKEN>', attr), templates.imagenet_templates_small))
    else:
        if args.with_shift:
            prompts += list(map(lambda s: prompt_template+s, SHIFTS))
        else:
            prompts += list(map(lambda s: s.format('<TOKEN>'), templates.imagenet_templates_small))

    imgs = []
    seed = 0
    num_per_batch = args.num_per_prompt * len(prompts)
    for c_id, c in enumerate(classes):
        print("Generating images for class {}".format(class_names[c_id]))
        for batch_id in range(args.num_gen // num_per_batch):
            print("Generating batch {}".format(batch_id))
            imgs_raw = generate(root, c, prompts, num_samples=args.num_per_prompt, random_seed=range(seed, seed+len(prompts)))
            imgs_lst = []
            for imgs in imgs_raw:
                imgs_lst += imgs
            seed += len(prompts)
            imgs.append(imgs_lst)

            for j, image in enumerate(imgs_lst):
                image_path = os.path.join(output_dir, filename_pattern.format(class_names[c_id], j + batch_id * args.num_per_prompt))
                image.save(image_path)
        

def get_args():
    parser = argparse.ArgumentParser(description='Generate a file with random numbers')
    parser.add_argument('--num_gen', '-n', type=int, default=32,
                        help='The number of images to generate per class')
    parser.add_argument('--num_per_prompt', type=int, default=1,
                        help='The number of images to generate per prompt')
    parser.add_argument('--with_edit', action='store_false',
                        help='Whether to use attribute edits')
    parser.add_argument('--with_shift', action='store_true',
                        help='Whether to use image shifts')
    parser.add_argument('--encoder_root_path', type=str, default='./encoder_root_imagenet',
                        help='The path to the encoder root directory')
    parser.add_argument('--output_dir', type=str, default='./generations',
                        help='The path to the output directory')
    return parser.parse_args()

def main():
    args = get_args()
    # gen_dataset(args, 'cat')
    gen_dataset(args, 'dog')

if __name__ == '__main__':
    main()