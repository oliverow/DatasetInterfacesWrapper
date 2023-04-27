# DatasetInterfacesWrapper
A wrapper for Dataset Interfaces repository for 2023 Spring CMU PGM course project

## Preparation
Install python environment with 
`conda create -f environment.yaml`

Follow `dataset_interfaces/notebooks/Example.ipynb` to prepare `encoder_root_imagenet` which contains necessary tokens and embeddings.

## Usage
To generate with the original Dataset Interfaces work,

`python gen.py --encoder_root_path <encoder_root_path>`

To generate with Dataset Interfaces and our discovered attributes,

`python gen.py --encoder_root_path <encoder_root_path> --with_edit`

To generate with Dataset Interfaces and diversity shifts,

`python gen.py --encoder_root_path <encoder_root_path> --with_shift`

Run `python gen.py -h` for a full usage description of the wrapper.

The total number of generated images is `args.num_per_prompt`$\times$`number of attributes provided`$\times$`number of prompt templates`.

The generated images are used as supplemental training data to improve the model.
