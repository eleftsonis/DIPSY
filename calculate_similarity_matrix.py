import argparse
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import clip
import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and store CLIP ViT-B/16 pairwise class similarity matrices."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. eurosat).")
    parser.add_argument(
        "--seed",
        default="seed16",
        help="Few-shot split identifier (default: seed16).",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory containing <dataset>/real_train_fewshot/<seed>/.",
    )
    parser.add_argument(
        "--clip-cache",
        default=None,
        help="Optional directory for caching CLIP weights.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save the similarity matrix (defaults to dataset path).",
    )
    return parser.parse_args()


def compute_similarity_matrix(
    dataset_path: Path, device: str, clip_cache: Optional[str]
) -> Tuple[np.ndarray, List[str]]:
    """Compute pairwise cosine similarities between class mean CLIP embeddings.

    For each class, all few-shot images are encoded and averaged to produce a
    single class embedding. The resulting N x N matrix is used by manage.py to
    sample negative classes proportionally to inter-class similarity.
    """
    model, preprocess = clip.load(
        "ViT-B/16", device=device, download_root=clip_cache or None
    )

    class_embeddings = {}
    for class_name in sorted(os.listdir(dataset_path)):
        class_folder = dataset_path / class_name
        if not class_folder.is_dir():
            continue

        embeddings = []
        for img_name in os.listdir(class_folder):
            img_path = class_folder / img_name
            if not img_path.is_file():
                continue
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image).cpu().numpy().flatten()
            embeddings.append(embedding)

        if not embeddings:
            raise ValueError(f"No images found in {class_folder}")
        class_embeddings[class_name] = np.mean(embeddings, axis=0)

    class_names = list(class_embeddings.keys())
    embedding_matrix = np.array([class_embeddings[c] for c in class_names])
    similarity_matrix = cosine_similarity(embedding_matrix)
    return similarity_matrix, class_names


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path = (
        Path(args.data_root) / args.dataset / "real_train_fewshot" / args.seed
    ).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    print(
        f"Computing similarity matrix for '{args.dataset}' "
        f"(seed: {args.seed}) from {dataset_path}"
    )
    start = time.time()
    similarity_matrix, class_names = compute_similarity_matrix(
        dataset_path, device, args.clip_cache
    )
    print(f"Done in {time.time() - start:.1f}s on {device}.")

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else dataset_path
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"similarity_matrix_{args.dataset}_{args.seed}.npz"
    np.savez(
        save_path,
        similarity_matrix=similarity_matrix,
        class_names=np.array(class_names),
    )
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
