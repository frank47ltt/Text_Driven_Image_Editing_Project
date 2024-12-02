import os

# Input data path (original shared folder)
BASE_PATH = "/content/drive/MyDrive/PIE-Bench_v1"
MAPPING_FILE = os.path.join(BASE_PATH, "mapping_file.json")
SOURCE_IMAGES_DIR = os.path.join(BASE_PATH, "annotation_images")

# Output paths (your personal folder)
OUTPUT_BASE_DIR = "/content/drive/MyDrive/ESE_645_result_sd_ddim_default"
GENERATED_IMAGES_DIR = os.path.join(OUTPUT_BASE_DIR, "generated_images")
EVALUATION_RESULTS_FILE = os.path.join(OUTPUT_BASE_DIR, "evaluation_results.csv")

# Categories
CATEGORIES = [
    "0_random_140",
    "1_change_object_80",
    "2_add_object_80",
    "3_delete_object_80",
    "4_change_attribute_content_40",
    "5_change_attribute_pose_40",
    "6_change_attribute_color_40",
    "7_change_attribute_material_40",
    "8_change_background_80",
    "9_change_style_80"
]

# Model configuration
DEFAULT_CONFIG = {
    "num_steps": 100,
    "start_step": 30,
    "guidance_scale": 3.5,
    "device": "cuda"
}
