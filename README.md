# RAG techniques for generating image captions

Phase One dataset exploration and preparation tools for the MS-COCO image
captioning project. The scripts do not download datasets. They audit the local
COCO images and annotations, create reproducible splits, preprocess images and
captions, generate embeddings, build an exact cosine index, and report quality
checks.

## Setup

From the project root:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-embeddings.txt
```

For part-of-speech statistics, install the optional analysis dependency and
English model:

```powershell
python -m pip install -r requirements-analysis.txt
python -m spacy download en_core_web_sm
```

## Phase One workflow

Update the paths for your COCO download and subset. `instances` annotations
provide image/category/object information; `captions` annotations provide
human-written captions.

```powershell
python scripts\phase1.py explore `
  --instances data\annotations\instances_train2017.json `
  --captions data\annotations\captions_train2017.json `
  --spacy-model en_core_web_sm `
  --output reports\exploration

python scripts\phase1.py clean `
  --images data\train2017 `
  --instances data\annotations\instances_train2017.json `
  --captions data\annotations\captions_train2017.json `
  --output reports\cleaning

python scripts\phase1.py split `
  --manifest reports\cleaning\clean_manifest.csv `
  --test-images data\external_test `
  --output reports\splits

python scripts\phase1.py preprocess `
  --splits reports\splits `
  --instances data\annotations\instances_train2017.json `
  --test-instances data\annotations\instances_val2017.json `
  --captions data\annotations\captions_train2017.json `
  --validation-captions data\annotations\captions_train2017.json `
  --test-captions data\annotations\captions_val2017.json `
  --output reports\prepared

python scripts\phase1.py embed `
  --manifest reports\prepared\train.csv `
  --captions reports\prepared\train_captions.csv `
  --output reports\embeddings\train_clip.npz

python scripts\phase1.py build-index `
  --embeddings reports\embeddings\train_clip.npz `
  --metadata reports\prepared\metadata.json `
  --output reports\index\train_flat.npz

python scripts\phase1.py quality-check `
  --manifest reports\prepared\train.csv `
  --embeddings reports\embeddings\train_clip.npz `
  --captions reports\prepared\train_captions.csv `
  --splits reports\splits `
  --metadata reports\prepared\metadata.json `
  --output reports\quality\train.json
```

`explore` writes dataset summaries, top-50 object classes, object
co-occurrence, available relationship/attribute/context distributions,
caption vocabulary and histograms. POS statistics are produced when
`--spacy-model` is supplied.

`clean` validates image files, removes exact byte duplicates from the clean
manifest (retaining the first image ID), records missing/corrupt/duplicate
files, and optionally audits the expected five captions per image. It never
deletes source files. Caption-count mismatches are documented rather than
silently discarded.

`split` defaults to 10,000 clean knowledge-base images, a reproducible
100-image validation subset, and 100 external test images (seed 42). The PDF's
dataset section says 100 test images, while its preprocessing checklist says
1,000; this workflow uses 100 as selected for this project. Override
`--test-count` if the agreed subset changes. `validation.csv` is a subset of
the 10,000-image knowledge base; `train.csv` excludes validation to prevent
self-retrieval. Test IDs and content are checked for overlap. Versioned
`split_ids.json` and `splits.json` record the split membership and leakage
audit.

`preprocess` writes EXIF-corrected RGB images, preserving aspect ratio, split
manifests, separate train/validation/test caption files, object and available
attribute/relation/context metadata, and a preprocessing report. Raw caption
text is retained unchanged; only the retrieval field is lowercased and
whitespace-normalized. Validation and test references are kept out of the
training caption file. Encoder-specific resizing and normalization are applied
by the selected encoder processor.

`embed` writes normalized CLIP image and text vectors, paired image/caption
IDs, raw and cleaned captions, and model, dimension, batch-size, device, and
version metadata. It accepts a COCO captions JSON or prepared caption CSV.
The default model is `openai/clip-vit-base-patch32`; Transformers downloads its
weights on first use.

`build-index` creates a reproducible exact flat cosine index containing image
and text vectors linked to image IDs, captions, and metadata. `quality-check`
checks duplicate IDs, missing/unexpected embeddings, non-finite or
unnormalized vectors, caption mismatches, metadata coverage, and split leakage.

Run the tests from the project root:

```powershell
python -m unittest discover -s tests
```
