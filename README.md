# DIPSY: Training-Free Synthetic Data Generation with Dual IP-Adapter Guidance

**BMVC 2025**

Luc Boudier\*,
Loris Manganelli\*,
[Eleftherios Tsonis](https://eleftsonis.github.io/)\*,
[Nicolas Dufour](https://nicolas-dufour.github.io/),
[Vicky Kalogeiton](https://vicky.kalogeiton.info/)

\* Equal contribution

[[Paper (arXiv:2509.22635)]](https://arxiv.org/abs/2509.22635) | [[Project Page]](https://www.lix.polytechnique.fr/vista/projects/2025_bmvc_dipsy/)

---

![Qualitative comparison of synthetic image generation for visually similar class pairs across datasets: British Shorthair vs Russian Blue (Pets), Risotto vs Paella (Food101), and Boeing 747-400 vs 777-300 (FGVC Aircraft). DIPSY generates semantically faithful and visually distinct images, preserving class-specific cues such as eye color in pets, food-specific textures and toppings, and structural aircraft details. Competing methods (DISEF and DataDream) often produce ambiguous results. Real images included for reference.](https://www.lix.polytechnique.fr/vista/projects/2025_bmvc_dipsy/static/images/qualitative_comparison_6_hq.png)

*Qualitative comparison of synthetic image generation for visually similar class pairs across datasets: British Shorthair vs Russian Blue (Pets), Risotto vs Paella (Food101), and Boeing 747-400 vs 777-300 (FGVC Aircraft). DIPSY generates semantically faithful and visually distinct images, preserving class-specific cues such as eye color in pets, food-specific textures and toppings, and structural aircraft details. Competing methods (DISEF and DataDream) often produce ambiguous results. Real images included for reference.*

---

## Overview

DIPSY (**D**ual **IP**-Adapter **Sy**nthesizer) is a training-free method for generating synthetic training data for few-shot image classification. Given only a handful of labeled real images per class, DIPSY uses two IP-Adapters simultaneously with Stable Diffusion to produce class-discriminative synthetic images:

- A **positive** image prompt from the target class guides the generation towards class-specific features.
- A **negative** image prompt from a visually similar class pushes the generation away from confusable features.

The negative class is sampled from a CLIP-based inter-class similarity distribution, so that the most challenging confounders are suppressed most strongly.

DIPSY requires no model fine-tuning, no external captioning, and no image filtering. It achieves state-of-the-art or competitive performance on 10 few-shot classification benchmarks.

---

## Method Summary

The pipeline has three stages:

1. **Similarity matrix** — Compute pairwise CLIP ViT-B/16 cosine similarities between classes from the few-shot images. This defines a probability distribution for sampling negative classes.

2. **Synthetic image generation** — For each target class, generate `n_imgs_per_class` synthetic images using the dual IP-Adapter guidance scheme:

   ```
   ε̂ = ε_uncond
       + w_text  · (ε_text         − ε_uncond)
       + w_im+   · (ε_text,im+     − ε_text)
       + w_im−   · (ε_text,im+,im− − ε_text,im+)
   ```

3. **Classifier fine-tuning** — Fine-tune a CLIP ViT-B/16 classifier with LoRA on the combined real and synthetic images, following the training setup of [DataDream](https://github.com/ExplainableML/DataDream). See [Classification](#classification) below.

---

## Dependencies

### Python packages

```
torch>=2.0
diffusers>=0.27
transformers
clip @ git+https://github.com/openai/CLIP.git
scipy
scikit-learn
Pillow
fire
numpy
```

Install with:

```bash
pip install torch diffusers transformers scipy scikit-learn Pillow fire numpy
pip install git+https://github.com/openai/CLIP.git
```

### Pre-trained model weights

- **Stable Diffusion 1.5**.
- **IP-Adapter Plus (SD 1.5)** — download `ip-adapter-plus_sd15.safetensors` from [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter). Place it under `<ip_adapter_path>/models/ip-adapter-plus_sd15.safetensors`.

---

## Data Format

Few-shot images should be organized as:

```
<data_root>/
  <dataset_name>/
    real_train_fewshot/
      <seed>/              # e.g. seed16
        <class_A>/
          image_001.jpg
          image_002.jpg
          ...
        <class_B>/
          ...
```

The same layout is used by the DataDream codebase for the classification step.

---

## Reproducing the Paper Results

### Step 1 — Compute the class similarity matrix

```bash
python calculate_similarity_matrix.py \
    --dataset <dataset_name> \
    --seed seed16 \
    --data-root /path/to/data \
    --output-dir /path/to/data/<dataset_name>/real_train_fewshot/seed16
```

This saves `similarity_matrix_<dataset_name>_seed16.npz` inside the seed directory. `manage.py` expects it there by default.

Supported dataset names: `dtd`, `food101`, `pets`, `sun397`, `eurosat`, `fgvc_aircraft`, `imagenet`, `flowers102`, `cars`, `caltech101`.

### Step 2 — Generate synthetic images

```bash
python manage.py \
    --dataset_name <dataset_name> \
    --seed_num seed16 \
    --data_root /path/to/data \
    --output_root /path/to/generated \
    --sd_model_path /path/to/stable-diffusion-v1-5 \
    --ip_adapter_path /path/to/IP-Adapter \
    --guidance_text <w_text> \
    --guidance_image_one <w_im+> \
    --guidance_image_two <-w_im-> \
    --n_imgs_per_class 200 \
    --num_inference_steps 50
```

**Important:** `guidance_image_two` must be a **negative** float to implement the paper's negative guidance. For example, if the paper uses `w_im− = 6`, pass `--guidance_image_two -6`.

To distribute generation across multiple GPUs (one process per GPU, slicing classes):

```bash
# GPU 0 of 4
python manage.py ... --num_jobs 4 --task_id 0

# GPU 1 of 4
python manage.py ... --num_jobs 4 --task_id 1
# etc.
```

Generated images are saved under:
```
<output_root>/<dataset_name>/seed16/<class_name>/negative_XXXX.jpg
```

### Step 3 — Train the classifier

The classification step follows the [DataDream](https://github.com/ExplainableML/DataDream) training pipeline. Please refer to that repository and apply it to the synthetic images generated in Step 2, alongside the original few-shot real images.

---

## Citation

If you use DIPSY in your research, please cite:

```bibtex
@inproceedings{boudier2025dipsy,
  title     = {Training-Free Synthetic Data Generation with Dual IP-Adapter Guidance},
  author    = {Boudier, Luc and Manganelli, Loris and Tsonis, Eleftherios and Dufour, Nicolas and Kalogeiton, Vicky},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2025}
}
```

---

## Acknowledgements

This project was partially supported by a Hi!Paris collaborative project, the Agence de l’Innovation de Défense – AID - via Centre Interdisciplinaire d’Etudes pour la Défense et la Sécurité – CIEDS - (project 2024 - FakeDetect) and was granted access to the HPC resources of IDRIS under the allocation 2024-AD011015893 made by GENCI. 