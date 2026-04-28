import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import fire
import numpy as np
import scipy.special
import torch
from PIL import Image
from diffusers import DPMSolverMultistepScheduler
from IPGen.pipeline_stable_diffusion_guidance_scheduler import (
    StableDiffusionPipelineAlternativeGuidance,
)

PIPELINE: Optional[StableDiffusionPipelineAlternativeGuidance] = None

DATASET_PROMPTS: Dict[str, str] = {
    "fgvc_aircraft": "A photo of a {class_name} aircraft",
    "eurosat": "A satellite photo of a {class_name}",
    "sun397": "A photo of a {class_name}",
    "flowers102": "A photo of a {class_name} flower",
    "dtd": "A photo of a {class_name} texture",
    "cars": "A photo of a {class_name} car",
    "food101": "A photo of a {class_name} food",
    "pets": "A photo of a {class_name} pet",
    "imagenet": "A photo of a {class_name}",
    "imagenet_100": "A photo of a {class_name}",
    "caltech101": "A photo of a {class_name}",
}
DEFAULT_PROMPT = "A photo of a {class_name}"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _ensure_pipeline(sd_model_path: str, device: str = "cuda"):
    global PIPELINE
    if PIPELINE is None:
        if not sd_model_path:
            raise ValueError("sd_model_path must be provided to load the pipeline.")
        PIPELINE = (
            StableDiffusionPipelineAlternativeGuidance.from_pretrained(
                sd_model_path, torch_dtype=torch.float16, safety_checker=None
            ).to(device)
        )
        PIPELINE.scheduler = DPMSolverMultistepScheduler.from_config(
            PIPELINE.scheduler.config
        )
    return PIPELINE


def _prompt_for_class(dataset_name: str, class_name: str) -> str:
    template = DATASET_PROMPTS.get(dataset_name, DEFAULT_PROMPT)
    return template.format(class_name=class_name)


def _list_images(folder: Path) -> List[Path]:
    return [
        folder / f
        for f in os.listdir(folder)
        if (folder / f).suffix.lower() in IMAGE_EXTENSIONS
    ]


def _augment_image(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    width, height = img.size
    if width != height:
        size = min(width, height)
        left = random.randint(0, width - size)
        top = random.randint(0, height - size)
        img = img.crop((left, top, left + size, top + size))

    angle = random.uniform(-20, 20)
    img = img.rotate(angle, expand=False)
    width, height = img.size
    max_side = max(width, height)
    size = int(max_side / (2 ** 0.5))
    left = (width - size) // 2
    top = (height - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return img


def _load_adapter_image(path: Path, apply_aug: bool) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if apply_aug:
        return _augment_image(image)
    return image


def _load_similarity(dataset_path: Path, dataset_name: str, seed_num: str):
    sim_path = dataset_path / f"similarity_matrix_{dataset_name}_{seed_num}.npz"
    if not sim_path.exists():
        raise FileNotFoundError(f"Similarity matrix not found at {sim_path}")
    data = np.load(sim_path)
    similarity_matrix = data["similarity_matrix"]
    class_names = data["class_names"].tolist()
    return similarity_matrix, class_names


def _build_probability_matrix(similarity_matrix: np.ndarray, temperature: float):
    prob_matrix = scipy.special.softmax(similarity_matrix / temperature, axis=1)
    np.fill_diagonal(prob_matrix, 0)
    prob_matrix /= prob_matrix.sum(axis=1, keepdims=True)
    return prob_matrix


def _select_negative_class(
    current_class: str,
    class_names: Sequence[str],
    prob_matrix: np.ndarray,
    class_to_index: Dict[str, int],
) -> str:
    idx = class_to_index[current_class]
    probs = prob_matrix[idx]
    return np.random.choice(class_names, p=probs)


def _load_ip_adapters(
    pipeline: StableDiffusionPipelineAlternativeGuidance,
    adapter_path: str,
    weight_names: Sequence[str],
):
    pipeline.load_ip_adapter(
        adapter_path, subfolder="models", weight_name=list(weight_names)
    )


def _run_generation(
    prompt: str,
    image_one: Image.Image,
    image_two: Image.Image,
    num_inference_steps: int,
    guidance_text: float,
    guidance_image_one: float,
    guidance_image_two: float,
    ip_adapter_scale: float,
):
    if PIPELINE is None:
        raise RuntimeError("Pipeline is not initialized.")
    generator = torch.Generator(device="cpu").manual_seed(
        random.randint(1, 5_000_000)
    )
    result = PIPELINE(
        prompt=prompt,
        ip_adapter_image=[image_one, image_two],
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_text=guidance_text,
        guidance_image_one=guidance_image_one,
        guidance_image_two=guidance_image_two,
        ip_adapter_scale=ip_adapter_scale,
    )
    return result.images[0]


def generate_images(
    dataset_name: str,
    seed_num: str = "seed0",
    n_imgs_per_class: int = 200,
    num_jobs: int = 1,
    task_id: int = 0,
    data_root: str = "./data",
    output_root: str = "./generated",
    sd_model_path: str = "",
    ip_adapter_path: str = "",
    ip_adapter_weights: Optional[Sequence[str]] = None,
    # Guidance scales. guidance_image_two must be negative to repel the generated
    # image from the negative class (e.g. -6.0 implements -w_im- from the paper).
    guidance_text: float = 7.5,
    guidance_image_one: float = 6.0,
    guidance_image_two: float = -6.0,
    num_inference_steps: int = 50,
    similarity_temperature: float = 0.4,
    use_class_similarity: bool = True,
    augment_inputs: bool = True,
    ip_adapter_scale: float = 0.6,
):
    """Generate synthetic few-shot images using DIPSY's dual IP-Adapter guidance.

    For each class in the dataset, generates n_imgs_per_class synthetic images by
    conditioning on a positive image from that class and a negative image from a
    visually similar class (selected via CLIP cosine similarity).

    The similarity matrix (.npz) must be pre-computed with calculate_similarity_matrix.py
    and placed inside <data_root>/<dataset_name>/real_train_fewshot/<seed_num>/.
    """
    dataset_path = (
        Path(data_root) / dataset_name / "real_train_fewshot" / seed_num
    ).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    output_base_dir = (
        Path(output_root) / dataset_name / seed_num
    ).expanduser()
    output_base_dir.mkdir(parents=True, exist_ok=True)

    class_folders = [
        d for d in sorted(os.listdir(dataset_path)) if (dataset_path / d).is_dir()
    ]
    if not class_folders:
        raise ValueError(f"No class folders found under {dataset_path}")

    num_classes = len(class_folders)
    classes_per_job = max(1, (num_classes + num_jobs - 1) // num_jobs)
    start_index = task_id * classes_per_job
    end_index = min(start_index + classes_per_job, num_classes)
    sub_class_folders = class_folders[start_index:end_index]
    if not sub_class_folders:
        print(f"No classes assigned to task {task_id} (num_jobs={num_jobs}).")
        return

    pipeline = _ensure_pipeline(sd_model_path)
    if not ip_adapter_path:
        raise ValueError("ip_adapter_path is required.")
    adapter_weights = (
        ip_adapter_weights
        if ip_adapter_weights is not None
        else ("ip-adapter-plus_sd15.safetensors", "ip-adapter-plus_sd15.safetensors")
    )
    if len(adapter_weights) != 2:
        raise ValueError("ip_adapter_weights must contain exactly two entries.")

    prob_matrix = None
    class_names: Sequence[str] = []
    class_to_index: Dict[str, int] = {}
    if use_class_similarity:
        similarity_matrix, class_names = _load_similarity(
            dataset_path, dataset_name, seed_num
        )
        class_to_index = {name: idx for idx, name in enumerate(class_names)}
        prob_matrix = _build_probability_matrix(similarity_matrix, similarity_temperature)

    for class_folder in sub_class_folders:
        class_dir = dataset_path / class_folder
        image_paths = _list_images(class_dir)
        if not image_paths:
            print(f"Skipping {class_folder}: no images found.")
            continue

        _load_ip_adapters(pipeline, ip_adapter_path, adapter_weights)
        class_output_dir = output_base_dir / class_folder
        class_output_dir.mkdir(parents=True, exist_ok=True)
        prompt = _prompt_for_class(dataset_name, class_folder)
        start = time.time()

        for idx in range(n_imgs_per_class):
            ref_one_path = random.choice(image_paths)

            if use_class_similarity and prob_matrix is not None:
                if class_folder not in class_to_index:
                    raise KeyError(
                        f"Class '{class_folder}' not found in similarity matrix."
                    )
                other_class = _select_negative_class(
                    class_folder, class_names, prob_matrix, class_to_index
                )
            else:
                candidates = [c for c in class_folders if c != class_folder]
                other_class = random.choice(candidates)

            other_dir = dataset_path / other_class
            other_images = _list_images(other_dir)
            if not other_images:
                print(f"Skipping image {idx}: no images in negative class '{other_class}'.")
                continue
            ref_two_path = random.choice(other_images)

            image_one = _load_adapter_image(ref_one_path, augment_inputs)
            image_two = _load_adapter_image(ref_two_path, augment_inputs)

            output_path = class_output_dir / f"negative_{idx:04d}.jpg"
            if output_path.exists():
                continue

            generated = _run_generation(
                prompt=prompt,
                image_one=image_one,
                image_two=image_two,
                num_inference_steps=num_inference_steps,
                guidance_text=guidance_text,
                guidance_image_one=guidance_image_one,
                guidance_image_two=guidance_image_two,
                ip_adapter_scale=ip_adapter_scale,
            )
            generated.save(output_path, "JPEG", quality=95)

        elapsed = time.time() - start
        print(f"Task {task_id} | {class_folder}: {elapsed:.1f}s")
        pipeline.unload_ip_adapter()
        pipeline.unload_ip_adapter()


if __name__ == "__main__":
    fire.Fire(generate_images)
