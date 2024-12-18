import os
import json
import abc
import torch
import torch.nn.functional as F
import torch.nn.functional as nnf
import numpy as np
from PIL import Image
from IPython.display import display
from tqdm import tqdm
from typing import Union, Tuple, List, Callable, Dict, Optional
from torch.optim.adam import Adam
from torchvision.transforms import ToTensor, Normalize

# Third-party imports
import clip
import lpips
from diffusers import StableDiffusionPipeline, DDIMScheduler
from skimage.metrics import structural_similarity

class Config:
    """Configuration class for the image editor"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_path = "data"
        self.output_path = "output"
        self.num_ddim_steps = 50
        self.guidance_scale = 7.5
        self.max_num_words = 77
        self.low_resource = False

class ModelManager:
    """Manages model initialization and cleanup"""
    def __init__(self, config: Config):
        self.config = config
        self.scheduler = self._init_scheduler()
        self.ldm_stable = self._init_stable_diffusion()
        self.tokenizer = self.ldm_stable.tokenizer

    def _init_scheduler(self):
        return DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False
        )

    def _init_stable_diffusion(self):
        model = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            scheduler=self.scheduler
        ).to(self.config.device)

        try:
            model.disable_xformers_memory_efficient_attention()
        except AttributeError:
            print("Attribute disable_xformers_memory_efficient_attention() is missing")

        return model

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'ldm_stable'):
            del self.ldm_stable
        torch.cuda.empty_cache()

class PTUtils:
    """Utility functions for prompt processing"""
    @staticmethod
    def get_word_inds(text: str, word_place: int, tokenizer) -> torch.Tensor:
        """Get indices of words in tokenized text"""
        words = text.split(' ')
        if word_place >= len(words):
            raise ValueError(f"Word place {word_place} is out of range for text: {text}")
        start_idx = len(tokenizer(' '.join(words[:word_place])).input_ids) - 1
        return torch.tensor([start_idx])

class LocalBlend:
    """Handles local blending of attention maps."""
    def __init__(self, prompts: List[str], words: List[List[str]], 
                 substruct_words=None, start_blend=0.2, th=(0.3, 0.3)):
        self.config = Config()
        self.ptp_utils = PTUtils()
        self.alpha_layers = self._initialize_alpha_layers(prompts, words)
        self.substruct_layers = self._initialize_substruct_layers(prompts, substruct_words)
        self.start_blend = int(start_blend * self.config.num_ddim_steps)
        self.counter = 0
        self.th = th

    def _initialize_alpha_layers(self, prompts, words):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, self.config.max_num_words)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if isinstance(words_, str):
                words_ = [words_]
            for word in words_:
                ind = self.ptp_utils.get_word_inds(prompt, word, self.tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        return alpha_layers.to(self.config.device)

    def _initialize_substruct_layers(self, prompts, substruct_words):
        if substruct_words is None:
            return None
            
        substruct_layers = torch.zeros(len(prompts), 1, 1, 1, 1, self.config.max_num_words)
        for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
            if isinstance(words_, str):
                words_ = [words_]
            for word in words_:
                ind = self.ptp_utils.get_word_inds(prompt, word, self.tokenizer)
                substruct_layers[i, :, :, :, :, ind] = 1
        return substruct_layers.to(self.config.device)

    def get_mask(self, maps, alpha, use_pool, x_t):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask

    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, self.config.max_num_words) 
                   for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True, x_t)
            
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False, x_t)
                mask = mask * maps_sub
                
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

class EmptyControl:
    """Empty control for baseline comparison."""
    def step_callback(self, x_t):
        return x_t
        
    def between_steps(self):
        return
        
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

class AttentionControl(abc.ABC):
    """Base class for attention control mechanisms."""
    def __init__(self, config: Config):
        self.config = config
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def step_callback(self, x_t):
        return x_t
        
    def between_steps(self):
        return
        
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.config.low_resource else 0
        
    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError
        
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.config.low_resource:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
        
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
class ImageEvaluator:
    """Evaluates image quality using multiple metrics"""
    def __init__(self, device: str):
        self.device = device
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

    def preprocess_image(self, image: Union[torch.Tensor, np.ndarray, Image.Image]) -> torch.Tensor:
        """Preprocesses images into normalized torch tensors"""
        if isinstance(image, Image.Image):
            image = np.array(image) / 255.0

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)

        image = (image * 2.0) - 1.0
        return image.to(self.device)

    def calculate_ssim(self, source: Union[torch.Tensor, np.ndarray],
                      target: Union[torch.Tensor, np.ndarray]) -> float:
        """Calculates Structural Similarity Index"""
        if isinstance(source, torch.Tensor):
            source = source.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        if source.ndim == 4:
            source = source[0]
        if target.ndim == 4:
            target = target[0]

        if source.shape[-1] == 3:
            source_gray = np.mean(source, axis=-1)
            target_gray = np.mean(target, axis=-1)
        else:
            source_gray = source
            target_gray = target

        return structural_similarity(source_gray, target_gray, data_range=1.0)

    def calculate_lpips(self, source: Union[torch.Tensor, np.ndarray],
                       target: Union[torch.Tensor, np.ndarray]) -> float:
        """Calculates LPIPS score"""
        source = self.preprocess_image(source)
        target = self.preprocess_image(target)

        with torch.no_grad():
            distance = self.lpips_model(source, target)
        return distance.item()

    def calculate_clip_similarity(self, image: Union[torch.Tensor, np.ndarray, Image.Image],
                                prompt: str) -> float:
        """Calculates CLIP similarity score"""
        if isinstance(image, (torch.Tensor, np.ndarray)):
            image = Image.fromarray(
                (image.cpu().numpy() * 255).astype(np.uint8) if isinstance(image, torch.Tensor)
                else (image * 255).astype(np.uint8)
            )

        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize([prompt]).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).item()

        return similarity

    def evaluate_all(self, source_image: Union[torch.Tensor, np.ndarray, Image.Image],
                    target_image: Union[torch.Tensor, np.ndarray, Image.Image],
                    target_prompt: str) -> Dict[str, float]:
        """Evaluates image using all available metrics"""
        metrics = {
            'ssim': self.calculate_ssim(source_image, target_image),
            'lpips': self.calculate_lpips(source_image, target_image),
            'clip_similarity': self.calculate_clip_similarity(target_image, target_prompt)
        }
        return metrics
    
class ImageEditor:
    """Handles image editing operations"""
    def __init__(self, config: Config, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.evaluator = ImageEvaluator(config.device)
        
    def _setup_controller(self, original_prompt: str, editing_prompt: str,
                         original_object: str, editing_object: str) -> AttentionControl:
        """Setup attention controller for image editing"""
        raise NotImplementedError("Implement specific attention control setup")

    def _generate_images(self, controller: AttentionControl) -> Tuple[List[Image.Image], Dict]:
        """Generate images using the model"""
        raise NotImplementedError("Implement image generation logic")

    def process_image_batch(self, data: Dict):
        """Process a batch of images with evaluation"""
        metrics_list = []
        
        for image_id, image_data in tqdm(data.items()):
            try:
                metrics = self._process_single_image(image_id, image_data)
                if metrics:
                    metrics_list.append(metrics)
            except Exception as e:
                print(f"Error processing image {image_id}: {str(e)}")
                continue
                
        return self._aggregate_metrics(metrics_list)
    
    def _process_single_image(self, image_id: str, image_data: Dict) -> Optional[Dict]:
        """Process a single image and return metrics"""
        image_path = os.path.join(self.config.data_path, image_data["image_path"])
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None

        original_prompt = image_data["original_prompt"]
        editing_prompt = image_data["editing_prompt"]
        
        # Extract objects from prompts
        original_object = original_prompt[original_prompt.find('[')+1:original_prompt.find(']')]
        editing_object = editing_prompt[editing_prompt.find('[')+1:editing_prompt.find(']')]

        # Process image
        controller = self._setup_controller(original_prompt, editing_prompt, original_object, editing_object)
        images, _ = self._generate_images(controller)
        
        # Evaluate results
        return self.evaluator.evaluate_all(
            source_image=images[0],
            target_image=images[1],
            target_prompt=editing_prompt
        )
    
    @staticmethod
    def _aggregate_metrics(metrics_list: List[Dict]) -> Dict:
        """Aggregate metrics from multiple images"""
        if not metrics_list:
            return {}
            
        aggregated = {}
        for metric in metrics_list[0].keys():
            values = [m[metric] for m in metrics_list]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        return aggregated

def main():
    """Main execution function"""
    config = Config()
    model_manager = ModelManager(config)
    
    try:
        # Load data
        with open('mapping_file.json', 'r') as f:
            data = json.load(f)
        
        # Initialize image editor
        editor = ImageEditor(config, model_manager)
        
        # Process images and get metrics
        metrics = editor.process_image_batch(data)
        
        # Print results
        print("\nAggregated Metrics:")
        print("-" * 40)
        for metric, values in metrics.items():
            print(f"\n{metric}:")
            for stat, value in values.items():
                print(f"  {stat}: {value:.4f}")
                
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise
    
    finally:
        model_manager.cleanup()

if __name__ == "__main__":
    main()