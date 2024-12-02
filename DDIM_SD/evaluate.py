import os
import json
import numpy as np
from PIL import Image
import csv
import argparse
from tqdm import tqdm
from matrics_calculator import MetricsCalculator
from config import *

def mask_decode(encoded_mask, image_shape=[512,512]):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))
    
    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i+1], length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j] = 1
            
    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:] = 1
    mask_array[-1,:] = 1
    mask_array[:,0] = 1
    mask_array[:,-1] = 1
            
    return mask_array

def calculate_metric(metrics_calculator, metric, src_image, tgt_image, src_mask, tgt_mask, src_prompt, tgt_prompt):
    if metric == "psnr":
        return metrics_calculator.calculate_psnr(src_image, tgt_image, None, None)
    elif metric == "lpips":
        return metrics_calculator.calculate_lpips(src_image, tgt_image, None, None)
    elif metric == "mse":
        return metrics_calculator.calculate_mse(src_image, tgt_image, None, None)
    elif metric == "ssim":
        return metrics_calculator.calculate_ssim(src_image, tgt_image, None, None)
    elif metric == "clip_similarity_source_image":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt, None)
    elif metric == "clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, None)
    elif metric == "clip_similarity_target_image_edit_part":
        if tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, tgt_mask)
    elif metric == "structure_distance":
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, mask_pred=None, mask_gt=None, use_gpu = True)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def evaluate_category(category, source_dir, generated_dir, metrics_calculator, metrics, mapping_data):
    """Evaluate images for a specific category."""
    # Filter mapping data for this category
    category_data = {k: v for k, v in mapping_data.items() 
                    if category in v['image_path']}
    
    if not category_data:
        print(f"No images found for category: {category}")
        return []
    
    results = []
    
    for image_id, item in tqdm(category_data.items(), desc=f"Evaluating {category}"):
        try:
            # Get image paths maintaining full subfolder structure
            src_path = os.path.join(source_dir, item['image_path'])
            # Get relative path from source dir to maintain structure
            relative_path = os.path.relpath(src_path, source_dir)
            gen_path = os.path.join(generated_dir, relative_path)
            
            # Skip if files don't exist
            if not os.path.exists(src_path) or not os.path.exists(gen_path):
                print(f"Skipping {image_id}")
                print(f"Source path: {src_path}")
                print(f"Generated path: {gen_path}")
                continue
            
            # Load images
            src_image = Image.open(src_path).convert('RGB')
            gen_image = Image.open(gen_path).convert('RGB')
            
            # Get mask
            mask = mask_decode(item['mask'])
            mask = mask[:,:,np.newaxis].repeat([3], axis=2)
            
            # Get prompts
            original_prompt = item['original_prompt'].replace("[", "").replace("]", "")
            editing_prompt = item['editing_prompt'].replace("[", "").replace("]", "")
            
            # Calculate metrics
            evaluation_result = [image_id]
            for metric in metrics:
                result = calculate_metric(
                    metrics_calculator,
                    metric,
                    src_image,
                    gen_image,
                    mask,
                    mask,
                    original_prompt,
                    editing_prompt
                )
                evaluation_result.append(result)
            
            results.append(evaluation_result)
            
        except Exception as e:
            print(f"Error processing {image_id}: {str(e)}")
            continue
            
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate generated images for a specific category')
    parser.add_argument('--category', type=str, choices=CATEGORIES, 
                       help='Category to evaluate (e.g., "0_random_140")')
    parser.add_argument('--source_dir', type=str, default=SOURCE_IMAGES_DIR,
                       help='Directory containing source images')
    parser.add_argument('--generated_dir', type=str, default=GENERATED_IMAGES_DIR,
                       help='Directory containing generated images')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_BASE_DIR,
                       help='Directory to save evaluation results')
    args = parser.parse_args()
    
    if not args.category:
        print("Please specify a category using --category. Available categories:")
        for cat in CATEGORIES:
            print(f"  - {cat}")
        exit(1)
    
    print("\nUsing paths:")
    print(f"Source directory: {args.source_dir}")
    print(f"Generated images: {args.generated_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Define metrics
    metrics = [
        "psnr",
        "lpips",
        "mse",
        "ssim",
        "clip_similarity_source_image",
        "clip_similarity_target_image",
        "clip_similarity_target_image_edit_part",
        "structure_distance"
    ]
    
    # Create results directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define category-specific results file
    category_results_file = os.path.join(
        args.output_dir,
        f"evaluation_results_{args.category}.csv"
    )
    
    # Initialize metrics calculator
    metrics_calculator = MetricsCalculator(device="cuda")
    
    # Load mapping file
    with open(MAPPING_FILE, "r") as f:
        mapping_data = json.load(f)
    
    print(f"\nEvaluating category: {args.category}")
    
    # Create CSV headers
    headers = ["file_id"] + metrics
    
    # Evaluate category
    results = evaluate_category(
        args.category, 
        args.source_dir, 
        args.generated_dir,
        metrics_calculator, 
        metrics, 
        mapping_data
    )
    
    # Save results
    if results:
        with open(category_results_file, 'w', newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(headers)
            csv_writer.writerows(results)
        print(f"\nResults saved to: {category_results_file}")
        
        # Calculate and print summary statistics
        results_array = np.array([row[1:] for row in results])
        means = np.nanmean(results_array.astype(float), axis=0)
        stds = np.nanstd(results_array.astype(float), axis=0)
        
        print("\nSummary Statistics:")
        print("Metric            Mean      Std")
        print("-" * 40)
        for metric, mean, std in zip(metrics, means, stds):
            print(f"{metric:20s} {mean:8.4f} {std:8.4f}")
    else:
        print(f"No results generated for category: {args.category}")

if __name__ == "__main__":
    main()