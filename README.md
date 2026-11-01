# Image Captioning RAG — Phase One

Reproducible MS-COCO 2017 dataset auditing and preparation. The CLI is the
source of truth; notebooks and reports call or describe these commands rather
than reimplementing pipeline logic. Source images and annotations are never
modified. Generated data, vectors, indexes, and reports are ignored by Git.

## Environment and configuration

Use Python 3.10 or later. Install the core image and analysis dependencies:

```powershell
python -m pip install -r requirements.txt
```

For CLIP image/text embeddings and Sentence-BERT captions:

```powershell
python -m pip install -r requirements-embeddings.txt
```

For FAISS indexes, install `requirements-faiss.txt` where a compatible
`faiss-cpu` wheel is available. POS analysis is optional:

```powershell
python -m pip install -r requirements-analysis.txt
python -m spacy download en_core_web_sm
```

Edit [`configs/phase1.yaml`](configs/phase1.yaml) for dataset paths and counts.
The checked-in file is JSON-formatted YAML, so the default configuration also
works without an external YAML parser. Paths are relative to the repository
root; command-line values override the config.

The supplied local COCO layout is configured as a sibling of this repository:

```text
../train2017/train2017
../val2017/val2017
../annotations_trainval2017/annotations/
```

Change those paths for other Windows, Linux, Colab, or Kaggle layouts.

## Dataset policy

Defaults are seed `42`, a random 10,500-image train2017 candidate pool, 10,000
knowledge-base images, a separate 100-image train2017 validation set, and 100
COCO val2017 test images. The project PDF contains a 100-versus-1,000 test-size
discrepancy; 100 is the agreed default and counts are configurable.

Knowledge base, validation, and test IDs and SHA-256 contents must be pairwise
disjoint. The test selection is drawn from `instances_val2017.json`, matched to
`captions_val2017.json` and the val2017 directory—not an arbitrary folder.
Original caption strings are retained in independent split files. Five
references per image are expected; extra or missing captions are reported,
not deleted or fabricated.

## Run the selection and split stages

From the repository root, first audit the source annotations and image
directory, then select and validate the deterministic candidate pool:

```powershell
python scripts\phase1.py audit
python scripts\phase1.py select
```

Freeze the train/validation/test assignment using the COCO sources in the
configuration:

```powershell
python scripts\phase1.py split
```

This writes `01_splits/splits_v2.json`, per-split image and caption CSVs, and a
reserve manifest. It also writes `near_duplicate_review.csv` with cross-split
difference-hash matches at the configured similarity threshold. Review each
flagged pair and change its `review_status` from `pending` to `accepted` or
`rejected`; exact SHA-256 duplicates block freezing, while perceptual matches
are surfaced for human review. Frozen splits are not overwritten on rerun. To explicitly
create a new version, use:

```powershell
python scripts\phase1.py split --force-new-splits
```

The previous manifest and CSV files are archived under `01_splits/history/`.
To use a different config, pass `--config path/to/phase1.yaml`; individual CLI
options override configured values.

## Further Phase One stages

For KB-only EDA, point the command at the frozen KB manifest:

```powershell
python scripts\phase1.py explore `
  --instances ..\annotations_trainval2017\annotations\instances_train2017.json `
  --captions ..\annotations_trainval2017\annotations\captions_train2017.json `
  --manifest 01_splits\knowledge_base_manifest.csv `
  --output reports\phase1\exploration
```

Optional POS tagging requires the spaCy install/model above. COCO supplies
object categories and boxes, not semantic relationships or color/scene labels.
EDA labels category co-occurrence as annotation-derived and color/context
mentions as caption-derived, not as verified visual properties. Scene
complexity is reported as an object-count proxy.

Use the frozen manifests with `preprocess`, `embed`, `build-index`,
`build-faiss-index`, `encode-query`, `retrieve`, and `quality-check` as shown by
each command's `--help`. Preprocessing corrects EXIF orientation, converts to
RGB, preserves aspect ratio, and records dimensions. Encoder-specific image
normalization belongs to the encoder processor.

`embed --encoder clip` creates CLIP image/text vectors; `embed --encoder sbert`
creates Sentence-BERT caption vectors. `build-index` writes a local exact
cosine NPZ artifact. With compatible model packages and weights installed,
`build-faiss-index` creates independent CLIP-image, CLIP-caption, and
Sentence-BERT-caption FAISS indexes. `encode-query` produces vectors in the
matching space, and `retrieve` queries a selected index with optional
same-image exclusions. These model-backed stages have not been run as part of
the dependency-free synthetic tests; install the optional dependencies and
models before relying on real-data retrieval.

After running the available stages, record their provenance and execution
status with:

```powershell
python scripts\phase1.py release-manifest
```

The generated release manifest stays marked **incomplete** until embedding,
FAISS indexing, retrieval validation, and their quality checks have actually
been executed.

Run the dataset-only quality checks after preprocessing (this works without
model dependencies):

```powershell
python scripts\phase1.py quality-check `
  --splits 01_splits `
  --preprocessed reports\phase1\preprocessed `
  --output reports\phase1\quality_control\dataset_quality_report.json
```

See [REPORT.md](REPORT.md) for the real-data counts, verified checks, and
remaining unexecuted stages.

## Tests

The synthetic unit and integration suite does not download COCO:

```powershell
python -m unittest discover -s tests -v
```
