# Phase One Technical Report

## Status

**Partially complete; not a final V3.2 acceptance release.** Dataset selection,
split freezing, source-data analysis, image preprocessing, dataset quality
checks, and synthetic regression tests have been executed. Model embeddings,
FAISS indexes, and end-to-end retrieval validation have not been executed, so
no semantic retrieval results are claimed.

## Dataset and selection

The source is MS COCO 2017. The configured local source uses the official
train2017 and val2017 image directories and their corresponding instance and
caption annotation files. A full train2017 annotation audit found 118,287
unique image records, 860,001 object annotations, and 591,753 captions, with
no unknown image IDs or missing image files.

With random seed 42, the candidate-selection command sampled and validated
10,500 train2017 records. All 10,500 were readable and unique by SHA-256. The
frozen, version-2 split contains:

| Split | Images | Captions |
|---|---:|---:|
| Knowledge base | 10,000 | 50,025 |
| Validation | 100 | 500 |
| Test | 100 | 500 |
| Reserve candidates | 400 | — |

Knowledge-base and validation images are selected from train2017. Test images
are selected from val2017 and checked against its instance annotations,
caption annotations, and image directory. All three image-ID intersections
and cross-split SHA-256 intersections are empty. All 100 test images have
exactly five original references. Twenty-five selected knowledge-base images
have a sixth source caption; these extra references are retained rather than
discarded.

The frozen manifest records source annotation checksums, image manifests,
caption manifests, and per-file hashes. A configured difference-hash check
compared cross-split pairs at threshold 0.95 and flagged no pairs. Its review
CSV is still emitted so any flagged pairs can be marked `accepted` or
`rejected` without silently removing images.

## Exploration and caption analysis

Exploration was run over the frozen 10,000-image knowledge base:

- 73,456 object annotations across 80 categories.
- Mean 7.35 annotated objects per image; 102 images have no object boxes.
- 50,025 captions; mean caption length 10.46 words (median 10, range 5–46).
- Vocabulary size: 10,025 tokens under the implemented tokenizer.
- Scene complexity is presented only as an object-count proxy.

Category co-occurrence comes from COCO object labels. COCO does not provide
verified semantic relationships, visual attributes, or scene labels in these
annotation files. Context/color mentions are caption-derived descriptions,
not visual ground truth. Optional part-of-speech analysis was not requested.
Tables and plots are written under `reports/phase1/exploration/`.

## Cleaning and preprocessing

Candidate selection rejects unreadable images and exact duplicate file
contents. The frozen split command independently checks image IDs and SHA-256
hashes across the knowledge base, validation, and test sets. The preprocessing
command applies EXIF orientation, converts images to RGB, preserves aspect
ratio, and records dimensions and content hashes. Its default maximum size is
zero (no resize); outputs are encoded as JPEG at quality 95. Raw caption text
is retained alongside cleaned retrieval text. No source images or annotations
are modified.

Preprocessing completed for all 10,200 images. The dataset-only quality check
verified image existence, decodability, RGB mode, and output SHA-256 values;
metadata coverage and source-to-preprocessed caption mappings also passed.
Split counts, disjoint IDs, and all 500 test references passed.

## Retrieval architecture and quality control

The CLI supports CLIP image/text embeddings, Sentence-BERT caption embeddings,
separate FAISS indexes, query encoding, top-k retrieval, and ID exclusions.
These model-backed stages have not been run on the real dataset. Consequently,
embedding dimensions, FAISS vector counts, retrieval ordering, and independent
validation-query results remain unverified. The exact local NPZ index and
quality-check command have synthetic test coverage only.

The synthetic test suite contains 15 tests covering selection, split
freezing/versioning, caption integrity, preprocessing, metadata, vector
integrity, and retrieval behavior. The most recent run passed all 15 tests.
GitHub Actions has been configured but was not run in this environment.

## Limitations and next steps

- PyTorch, Transformers, Sentence Transformers, FAISS, and spaCy are not
  installed in the current environment. Install compatible embedding
  dependencies and model weights, then run embeddings, all three indexes,
  query encoding, and the 100-query validation protocol.
- Run embedding/index quality checks and record a technical retrieval smoke
  test separately from semantic relevance metrics.
- Optional spaCy part-of-speech analysis remains unexecuted.
- The pipeline is currently concentrated in `scripts/phase1.py`, rather than
  split into the proposed `src/` modules.
- Standard COCO annotations do not support verified visual attributes,
  semantic spatial relations, or scene labels; these must not be inferred
  from annotation absence.

Use the commands in [README.md](./README.md) to reproduce the available
stages. `release_manifest.json` records the execution state and must remain
marked incomplete until the unexecuted model and retrieval checks pass.
