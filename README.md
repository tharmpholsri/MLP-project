# Hierarchical Conditional Similarity Learning for Fine-Grained Visual Retrieval

---

## What is this project?

When you search for an image (e.g. a "Kettle"), a good retrieval system should return images that are visually similar at multiple levels — not just the exact same object, but also related items within the same broader category (e.g. "Appliances").

CLIP is a powerful pre-trained model that encodes images into vectors for retrieval. However, its embeddings become less reliable as you retrieve more results — precision drops from 95.8% at top-1 to 72.2% at top-1000. The root cause is geometric: CLIP's embedding space is too spread out and lacks hierarchical structure.

This project introduces **CLIP-CSN**, a lightweight extension of CLIP that adds:
- A **projection head** (small neural network) on top of frozen CLIP
- A **learned continuous mask** that highlights dimensions relevant to fine-grained subclass discrimination
- A **joint training objective** that enforces both superclass-level grouping and subclass-level separation

The result is a more compact, semantically organised embedding space that maintains high precision even at large retrieval depths.

---

## Models Compared

| Model | Description |
|-------|-------------|
| **Frozen CLIP** | Unmodified CLIP ViT-B/32 — zero-shot baseline |
| **Finetuned CLIP** | CLIP + projection head, trained with subclass contrastive loss (no mask) |
| **CLIP-CSN** | CLIP + projection head + learned hierarchical mask (this project) |

---

## Dataset

A custom fine-grained retrieval dataset of **64,723 images** spanning **24 subclasses** across **6 superclasses**:

- Appliances (Coffee Maker, Fan, Kettle, Lamp, Toaster)
- Fashion (Bottomwear, Topwear, Shoes)
- Furniture (Cabinet, Chair, Sofa, Table)
- Vehicles (Passenger Car, Truck, Van, Bicycle, Motorcycle)
- Tableware (Fork, Knife, Spoon, Mug)
- Stationery (Book, Pen, Stapler)

---

## Repository Structure

```
open_clip/
├── train_clip_csn.py                        # Train CLIP-CSN (projection head + mask)
├── train_clip_csn_nomask.py                 # Train Finetuned CLIP (projection head only)
├── generate_csn_embeddings_image_only.py    # Generate CLIP-CSN embeddings
├── generate_csn_embeddings_image_nomask.py  # Generate Finetuned CLIP embeddings
├── generate_clip_embeddings_image_nomask.py # Generate frozen CLIP baseline embeddings
├── csn_pipeline/
│   ├── model.py                             # ProjectionHead + SharedCSNMask architecture
│   ├── losses.py                            # Supervised contrastive loss functions
│   └── data.py                             # Dataset loading and train/test split
└── inference_pipeline/
    ├── run_csn_inference_image.py           # Retrieval evaluation: Precision@K, purity, error severity
    └── run_csn_inference.py                 # Full retrieval evaluation with geometric diagnostics
```

---

## Requirements

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm Pillow pandas scipy
pip install git+https://github.com/openai/CLIP.git
```

---

## How to Run

### Step 1 — Train

```bash
# Train CLIP-CSN
python train_clip_csn.py --data-csv /path/to/ALL_text_dataset-2.csv \
                         --image-root /path/to/dataset \
                         --output-dir ./training_output/csn

# Train Finetuned CLIP (no mask)
python train_clip_csn_nomask.py --data-csv /path/to/ALL_text_dataset-2.csv \
                                --image-root /path/to/dataset \
                                --output-dir ./training_output/nomask
```

### Step 2 — Generate Embeddings

```bash
# Frozen CLIP baseline
python generate_clip_embeddings_image_nomask.py

# Finetuned CLIP
python generate_csn_embeddings_image_nomask.py

# CLIP-CSN
python generate_csn_embeddings_image_only.py
```

### Step 3 — Evaluate Retrieval

```bash
python inference_pipeline/run_csn_inference_image.py \
    --embeddings /path/to/embeddings.npy \
    --category-ids /path/to/category_ids.npy \
    --superclass-ids /path/to/superclass_ids.npy
```

---

## Key Results

| Model | P@1 | P@10 | P@100 | P@1000 | Drop |
|-------|-----|------|-------|--------|------|
| Frozen CLIP | 95.8 | 93.2 | 91.8 | 72.2 | 23.6 |
| Finetuned CLIP | 95.5 | 95.3 | 95.5 | 94.3 | 1.2 |
| **CLIP-CSN** | **95.7** | **95.5** | **95.5** | **94.1** | **1.3** |

CLIP-CSN reduces cross-superclass neighbourhood contamination from **11.9% to 1.8%** compared to frozen CLIP.
