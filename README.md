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

Generated datasets, embeddings, and indexes may be stored in Google Drive rather than committed to GitHub because of their size.

## 5. Development Workflow

The project is divided into notebooks, with each notebook responsible for a specific stage.

| Notebook | Description |
|---|---|
| `00_workspace_setup.ipynb` | Mounts Google Drive and initializes the project workspace. |
| `01_dataset_audit.ipynb` | Audits dataset files, annotation records, and image availability. |
| `02_split_and_statistics.ipynb` | Creates dataset splits and performs statistical exploration. |
| `03_preprocessing.ipynb` | Handles image preprocessing and preparation. |
| `04_embeddings_and_index.ipynb` | Generates image and caption embeddings and builds FAISS indexes. |
| `05_quality_control.ipynb` | Performs consistency checks on generated artifacts. |
| `06_final_split.ipynb` | Contains an alternative final-split implementation that requires reconciliation with the canonical split. |

### Recommended execution order

For a clean execution, use the following sequence:

1. Workspace setup.
2. Dataset audit.
3. Split construction and statistical analysis.
4. Image preprocessing.
5. Embedding generation and FAISS index construction.
6. Final-scale quality control after its paths and expected counts have been aligned with the selected release.

**Before treating the project as a final release, resolve the competing split implementations and confirm the authoritative manifests.**

## 6. Technologies and Models

The project uses the following technologies and models.

| Technology / Model | Purpose |
|---|---|
| Python | Main programming language |
| Google Colab | Notebook execution environment |
| Google Drive | Persistent dataset and artifact storage |
| Git and GitHub | Version control and collaboration |
| Pandas | Dataset and annotation processing |
| NumPy | Numerical operations and embedding storage |
| PIL (Pillow) | Image loading and processing |
| Hugging Face Transformers | CLIP model integration |
| Sentence-Transformers | Text embedding generation |
| FAISS | Vector similarity search |

### Embedding models

**1. CLIP — Image Embeddings**

Model: `openai/clip-vit-base-patch32`

- Generates visual embeddings from images.
- Embedding dimension: 512.
- Used to represent images in the vector space.

**2. Sentence-BERT — Caption Embeddings**

Model: `sentence-transformers/all-MiniLM-L6-v2`

- Generates semantic embeddings from textual captions.
- Embedding dimension: 384.
- Used to represent captions for semantic similarity search.

### Vector indexing

The project uses FAISS with `IndexFlatIP`.

The embeddings are normalized, making inner-product similarity equivalent to cosine similarity.

## 7. Phase 1: Current Implementation Results

Phase 1 focuses on dataset preparation, knowledge-base construction, embedding generation, and vector indexing.

The final-scale embedding notebook reports the following results:

| Component | Result |
|---|---:|
| Knowledge-base images | 10,000 |
| Knowledge-base captions | 50,030 |
| Image embedding shape | `(10000, 512)` |
| Caption embedding shape | `(50030, 384)` |
| Image FAISS index size | 10,000 vectors |
| Caption FAISS index size | 50,030 vectors |

### Generated artifacts

The final embedding notebook records the following output structure:

```text
phase1_final/
`-- embeddings_and_index/
    |
    |-- captions/
    |   `-- knowledge_base_captions.parquet
    |
    |-- clip/
    |   |-- kb_image_embeddings.npy
    |   `-- kb_image_ids.npy
    |
    |-- sbert/
    |   `-- kb_caption_embeddings.npy
    |
    |-- index/
    |   |-- clip_image_index.faiss
    |   |-- clip_image_index_mapping.parquet
    |   |-- sbert_caption_index.faiss
    |   `-- sbert_caption_index_mapping.parquet
    |
    `-- embedding_metadata.json
```

These files represent the generated embeddings, their mappings, and the vector indexes.

### Statistical exploration

The dataset analysis also includes:

- Object frequency distributions.
- Object-count statistics.
- Caption-length distributions.
- Vocabulary analysis.
- Part-of-speech distributions.
- Frequent object co-occurrence pairs.

## 8. Execution Environment

The project has been developed primarily in Google Colab.

### Google Drive setup

The workspace is located at:

```python
/content/drive/MyDrive/image-captioning-rag
```

The Drive filesystem should be mounted before executing the notebooks.

Example:

```python
from google.colab import drive

drive.mount('/content/drive')
```

Define the project workspace:

```python
from pathlib import Path

WORKSPACE = Path(
    "/content/drive/MyDrive/image-captioning-rag"
)

assert WORKSPACE.exists(), "Project workspace not found"
```

### Important execution notes

- Google Colab runtimes are temporary. Mount Drive and rerun the required initialization cells after reconnecting.
- Generated artifacts should be saved to Google Drive to preserve them between sessions.
- GPU availability depends on the Colab runtime. The recorded final embedding run used CPU.
- The notebooks should be executed in the appropriate order to ensure that required manifests and artifacts exist.

## 9. Quality Control and Known Issues

Several checks have been implemented, but some items remain before the project can be considered fully validated.

### Completed or demonstrated

- Dataset inventory and image-availability auditing.
- Restoration of 4,908 missing selected knowledge-base images.
- Construction of a 10,000-image knowledge base.
- Generation of image and caption embeddings.
- Construction of image and caption FAISS indexes.
- Split-level ID disjointness checks in the canonical split implementation.
- Embedding and index smoke tests.

### Items requiring attention

1. **Conflicting split implementations:** `02_split_and_statistics.ipynb` and `06_final_split.ipynb` use different validation sources and produce different caption counts.
2. **Test-set size:** The specification must be clarified to determine whether the final test subset should contain 100 or 1,000 images.
3. **Caption count policy:** The team must decide whether to preserve all COCO captions or enforce exactly five captions per image.
4. **Final-scale quality control:** The existing QC output reports 1,000 images and 5,001 captions, so it does not yet validate the final 10,000-image release.
5. **Leakage verification:** The final image and caption indexes must be checked against the exact validation and test IDs and captions.
6. **Artifact verification:** Final binary artifacts should be checked for existence, integrity, and consistency in Google Drive.
7. **Evaluation:** A complete held-out caption-generation evaluation and corresponding metrics have not yet been demonstrated.

## 10. Project Status

**Current focus: Phase 1 — Knowledge-Base Construction**

The final-scale embedding notebook reports successful construction of the 10,000-image knowledge base and its corresponding vector indexes.

However, final release approval remains conditional on resolving the split and specification inconsistencies and completing quality control against the final-scale artifacts.

The subsequent retrieval, caption-generation, and evaluation stages remain part of the broader project objective.

## 11. Collaboration and Version Control

The project is maintained using Git and GitHub.

Repository:

https://github.com/djidji09/image-captioning-rag

When contributing:

1. Create or switch to an appropriate working branch.
2. Make changes in the relevant notebook, configuration, or documentation.
3. Test the changes before committing.
4. Use descriptive commit messages.
5. Push the branch to GitHub.
6. Merge changes into the main branch after review.

Avoid committing large datasets, generated embedding matrices, and FAISS index files unless the project explicitly requires them to be version-controlled. Store such artifacts in the designated persistent storage location and document how to access them.

## 12. Future Work

The broader project aims to extend the prepared knowledge base into a complete retrieval-augmented image captioning system.

Potential next steps include:

- Implementing image-based retrieval.
- Implementing caption-based semantic retrieval.
- Combining retrieved visual and textual information.
- Integrating a caption-generation model.
- Evaluating generated captions against held-out COCO references.
- Measuring caption quality using suitable metrics such as BLEU, METEOR, ROUGE-L, and CIDEr.
- Comparing the retrieval-augmented approach with a baseline captioning model.
- Improving reproducibility and final artifact validation.

---

**Project:** Image Captioning RAG  
**Dataset:** MS COCO 2017  
**Repository:** [image-captioning-rag](https://github.com/djidji09/image-captioning-rag)
