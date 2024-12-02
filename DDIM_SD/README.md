image_generate.py contains the pipeline of applying stable diffusion and DDIM scheduler.

image_edit.py imports functions from image_generate.py and access PIE_BENCH dataset to generate images.

matrics_calculator.py contains the evaluation metrics

evaluate.py imports functions from matrics_calculator.py

config.py contians the configurations

floder_utils.py is used to detect subfloders in each task category.
