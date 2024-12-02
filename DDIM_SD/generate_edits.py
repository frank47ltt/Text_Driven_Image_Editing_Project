import os
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from generate_image import edit, load_image
from config import *
from folder_utils import get_subfolder_structure

def generate_edited_images(
    mapping_file_path,
    source_images_dir,
    output_dir,
    category_path,
    device="cuda",
    num_steps=100,
    start_step=30,
    guidance_scale=3.5
):
    """Generate edited images for a specific category path."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(mapping_file_path, 'r') as f:
        mapping_data = json.load(f)
    
    # Filter by category path
    category_data = {k: v for k, v in mapping_data.items() 
                    if category_path in v['image_path']}
    
    if not category_data:
        print(f"No images found for path: {category_path}")
        return
    
    for image_id, item in tqdm(category_data.items(), desc=f"Processing {category_path}"):
        try:
            source_image_path = os.path.join(source_images_dir, item['image_path'])
            relative_path = os.path.relpath(source_image_path, source_images_dir)
            output_image_path = os.path.join(output_dir, relative_path)
            
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            
            if os.path.exists(output_image_path):
                print(f"Skipping {image_id} - already exists")
                continue
            
            source_image = load_image(source_image_path)
            original_prompt = item['original_prompt'].replace("[", "").replace("]", "")
            editing_prompt = item['editing_prompt'].replace("[", "").replace("]", "")
            
            print(f"\nProcessing {image_id}:")
            print(f"Source: {source_image_path}")
            print(f"Output: {output_image_path}")
            print(f"Original prompt: {original_prompt}")
            print(f"Editing prompt: {editing_prompt}")
            
            edited_image = edit(
                input_image=source_image,
                input_image_prompt=original_prompt,
                edit_prompt=editing_prompt,
                num_steps=num_steps,
                start_step=start_step,
                guidance_scale=guidance_scale
            )
            
            edited_image.save(output_image_path)
            print(f"Successfully processed {image_id}")
            
        except Exception as e:
            print(f"Error processing {image_id}: {str(e)}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate edited images for a category')
    parser.add_argument('--category', type=str, choices=CATEGORIES,
                       help='Main category (e.g., "1_change_object_80")')
    parser.add_argument('--all_subfolders', action='store_true',
                       help='Process all subfolders in the category')
    args = parser.parse_args()
    
    if not args.category:
        print("Please specify a category using --category.")
        exit(1)
    
    # Get all valid subfolder paths for this category
    subfolder_paths = get_subfolder_structure(SOURCE_IMAGES_DIR, args.category)
    
    if args.all_subfolders:
        # Process all subfolders
        for subfolder_path in subfolder_paths:
            print(f"\nProcessing subfolder: {subfolder_path}")
            
            category_output_dir = os.path.join(GENERATED_IMAGES_DIR, subfolder_path)
            
            config = {
                "mapping_file_path": MAPPING_FILE,
                "source_images_dir": SOURCE_IMAGES_DIR,
                "output_dir": GENERATED_IMAGES_DIR,
                "category_path": subfolder_path,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "num_steps": DEFAULT_CONFIG["num_steps"],
                "start_step": DEFAULT_CONFIG["start_step"],
                "guidance_scale": DEFAULT_CONFIG["guidance_scale"]
            }
            
            generate_edited_images(**config)
    else:
        # Process just the main category
        config = {
            "mapping_file_path": MAPPING_FILE,
            "source_images_dir": SOURCE_IMAGES_DIR,
            "output_dir": GENERATED_IMAGES_DIR,
            "category_path": args.category,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "num_steps": DEFAULT_CONFIG["num_steps"],
            "start_step": DEFAULT_CONFIG["start_step"],
            "guidance_scale": DEFAULT_CONFIG["guidance_scale"]
        }
        
        generate_edited_images(**config)
