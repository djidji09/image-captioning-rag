# Image Captioning RAG

**An Image Captioning System Using Retrieval-Augmented Generation (RAG)**

## 1. Project Overview

This project aims to develop an image captioning system that generates natural-language descriptions of images by combining visual understanding with information retrieved from a knowledge base.

The system uses **Retrieval-Augmented Generation (RAG)** to enrich image captioning. Instead of relying exclusively on a generative model, it is designed to retrieve relevant visual and textual information from a prepared knowledge base and use that information to support caption generation.

The project uses a subset of the **MS COCO 2017 dataset** for knowledge-base construction, validation, and testing.

### Main objectives

- Prepare and audit the MS COCO dataset.
- Build a structured image-caption knowledge base.
- Generate image and caption embeddings.
- Build vector indexes for similarity-based retrieval.
- Retrieve relevant images and captions to support the caption-generation process.
- Evaluate the quality and relevance of generated captions.

## 2. Project Architecture

The planned system follows this general workflow:

```text
             MS COCO 2017 Dataset
                       |
                       v
              Dataset Preparation
                       |
                       v
              Dataset Splitting
                       |
                       v
                Preprocessing
                       |
                       v
              Knowledge Base
                       |
             +---------+---------+
             |                   |
             v                   v
       Image Embeddings    Caption Embeddings
          (CLIP)              (Sentence-BERT)
             |                   |
             v                   v
       Image FAISS Index   Caption FAISS Index
             |                   |
             +---------+---------+
                       |
                       v
                Retrieval Stage
                       |
                       v
             Caption Generation
                       |
                       v
                Evaluation
```

*Note: This diagram represents the intended overall workflow. Phase 1 focuses primarily on dataset preparation, embedding generation, and vector-index construction.*

## 3. Dataset

The project uses the **MS COCO 2017 (Microsoft Common Objects in Context)** dataset.

COCO provides images, human-written captions, and object annotations that can be used for image understanding and captioning tasks.

### Dataset organization

The original dataset is organized as follows:

```text
00_original_data/
|
|-- train2017/
|   `-- Training images
|
|-- val2017/
|   `-- Validation images
|
`-- annotations/
    |-- captions_train2017.json
    |-- captions_val2017.json
    |-- instances_train2017.json
    `-- instances_val2017.json
```

### Intended dataset split

The canonical split currently recorded in `splits_v1.json` is:

| Subset | Number of images | Source |
|---|---:|---|
| Knowledge Base | 10,000 | COCO train2017 |
| Validation | 100 | COCO train2017 |
| Test | 100 | COCO val2017 |

The knowledge base is intended to remain separate from the validation and test subsets.

**Important:** The project specification contains an unresolved inconsistency: one section requests 100 test images, while another requests 1,000. The current canonical split uses 100 test images, following the 100-image requirement stated in the dataset-splitting sections.

The caption pipeline currently preserves the available COCO captions rather than enforcing exactly five captions per image. Consequently, the reported knowledge base contains 50,030 captions for 10,000 images.

## 4. Repository Structure

The main project structure is organized into notebooks, configuration files, data directories, and generated artifacts.

```text
image-captioning-rag/
|
|-- 00_original_data/
|   |-- train2017/
|   |-- val2017/
|   `-- annotations/
|
|-- 01_manifests/
|
|-- 01_splits/
|   `-- splits_v1.json
|
|-- 02_processed_data/
|
|-- 03_embeddings/
|
|-- 04_indexes/
|
|-- 06_releases/
|   `-- phase1_final/
|
|-- artifacts/
|
|-- reports/
|
|-- configs/
|   `-- phase1.yaml
|
|-- notebooks/
|   |-- 00_workspace_setup.ipynb
|   |-- 01_dataset_audit.ipynb
|   |-- 02_split_and_statistics.ipynb
|   |-- 03_preprocessing.ipynb
|   |-- 04_embeddings_and_index.ipynb
|   |-- 05_quality_control.ipynb
|   `-- 06_final_split.ipynb
|
`-- README.md
```

Generated datasets, embeddings, and indexes may be stored in Google Drive rather than committed to GitHub because of
