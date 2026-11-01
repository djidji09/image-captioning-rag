# RAG techniques for generating image captions

Phase 1 scripts for exploring and preparing an MS-COCO image-captioning dataset.
The scripts do not download data; run them after the image files and COCO
annotation JSON files are available.

## Setup

From the repository root, install the data exploration and image-validation
dependencies:

```powershell
python -m pip install -r requirements.txt
```

For optional CLIP image embeddings, also install:

```powershell
python -m pip install -r requirements-embeddings.txt
```

## Phase 1 workflow

The commands below assume the standard COCO annotation names. Replace paths
with the locations used for your downloaded subset. `instances` annotations
provide image/category/object information; `captions` annotations provide the
human-written captions.

```powershell
python scripts\phase1.py explore `
  --instances data\annotations\instances_train2017.json `
  --captions data\annotations\captions_train2017.json `
  --output reports\exploration

python scripts\phase1.py clean `
  --images data\train2017 `
  --instances data\annotations\instances_train2017.json `
  --output reports\cleaning `
  --copy-to data\cleaned_images

python scripts\phase1.py split `
  --manifest reports\cleaning\clean_manifest.csv `
  --test-images data\external_test `
  --output reports\splits

python scripts\phase1.py embed `
  --manifest reports\splits\train.csv `
  --captions data\annotations\captions_train2017.json `
  --output reports\embeddings\train_clip.npz
```

`explore` writes summary JSON, per-category and per-image CSV files, and
histograms for caption lengths, object counts, and captions per image.
`clean` checks files against their annotation records, removes exact byte
duplicates from the clean manifest (keeping the first image ID), and records
missing/corrupt/duplicate files in a report. It never deletes source files;
`--copy-to` optionally creates a physical copy of retained images.

By default, `split` requires exactly 10,000 clean knowledge-base images, 100
validation images, and exactly 100 valid images in the separate external test
directory. The random seed defaults to 42. The knowledge base records all
10,000 clean images; `validation.csv` is a subset of that pool and `train.csv`
contains the remaining images. Generate retrieval embeddings from `train.csv`
to avoid retrieving a validation image as its own example. Test image content
is checked against the knowledge base to prevent accidental overlap. If the
downloaded subset uses different counts, adjust `--expected-count`,
`--validation-count`, or `--test-count` explicitly.

The optional `embed` command writes normalized CLIP image vectors to an NPZ
file and model/count/dimension metadata to a neighboring JSON file. The default
encoder is `openai/clip-vit-base-patch32`; its weights are fetched by
Transformers the first time the command runs.

Run the lightweight script tests with:

```powershell
python -m unittest discover -s tests
```
