#!/usr/bin/env python3
"""Explore and prepare an MS-COCO image-captioning dataset."""

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def image_statistics(images, instances, captions):
    objects_by_image = Counter(
        annotation["image_id"] for annotation in instances
    )
    captions_by_image = defaultdict(list)
    for annotation in captions:
        captions_by_image[annotation["image_id"]].append(annotation["caption"])

    rows = []
    for image in sorted(images, key=lambda item: str(item["id"])):
        image_id = image["id"]
        image_captions = captions_by_image[image_id]
        lengths = [len(caption.split()) for caption in image_captions]
        rows.append(
            {
                "image_id": image_id,
                "file_name": image["file_name"],
                "object_count": objects_by_image[image_id],
                "caption_count": len(image_captions),
                "average_caption_words": (
                    sum(lengths) / len(lengths) if lengths else 0
                ),
                "maximum_caption_words": max(lengths, default=0),
            }
        )
    return rows


def explore(args):
    instance_data = load_json(args.instances)
    caption_data = load_json(args.captions)
    images = instance_data.get("images") or caption_data.get("images", [])
    if not images:
        raise ValueError("No image records were found in the COCO annotations.")

    instances = instance_data.get("annotations", [])
    captions = caption_data.get("annotations", [])
    categories = {
        category["id"]: category["name"]
        for category in instance_data.get("categories", [])
    }

    category_counts = Counter(
        categories.get(annotation["category_id"], str(annotation["category_id"]))
        for annotation in instances
    )
    rows = image_statistics(images, instances, captions)
    caption_lengths = [
        len(annotation["caption"].split()) for annotation in captions
    ]
    object_counts = [row["object_count"] for row in rows]
    caption_counts = [row["caption_count"] for row in rows]

    summary = {
        "image_count": len(images),
        "object_annotation_count": len(instances),
        "caption_count": len(captions),
        "category_count": len(category_counts),
        "mean_objects_per_image": (
            sum(object_counts) / len(object_counts) if object_counts else 0
        ),
        "mean_captions_per_image": (
            sum(caption_counts) / len(caption_counts) if caption_counts else 0
        ),
        "mean_caption_words": (
            sum(caption_lengths) / len(caption_lengths)
            if caption_lengths
            else 0
        ),
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    write_csv(
        output / "categories.csv",
        [
            {"category": category, "object_count": count}
            for category, count in sorted(category_counts.items())
        ],
        ["category", "object_count"],
    )
    write_csv(
        output / "image_statistics.csv",
        rows,
        [
            "image_id",
            "file_name",
            "object_count",
            "caption_count",
            "average_caption_words",
            "maximum_caption_words",
        ],
    )
    create_histograms(output, caption_lengths, object_counts, caption_counts)
    print(f"Exploration reports written to {output.resolve()}")


def create_histograms(output, caption_lengths, object_counts, caption_counts):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Histogram generation requires matplotlib. Install requirements.txt."
        ) from error

    for values, title, xlabel, filename in (
        (caption_lengths, "Caption length distribution", "Words per caption", "caption_lengths.png"),
        (object_counts, "Objects per image", "Object annotations", "objects_per_image.png"),
        (caption_counts, "Captions per image", "Human-written captions", "captions_per_image.png"),
    ):
        figure, axis = plt.subplots()
        axis.hist(values, bins="auto")
        axis.set(title=title, xlabel=xlabel, ylabel="Count")
        figure.tight_layout()
        figure.savefig(output / filename)
        plt.close(figure)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path):
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Image validation requires Pillow. Install requirements.txt."
        ) from error

    with Image.open(path) as image:
        image.verify()


def clean(args):
    instance_data = load_json(args.instances)
    images = instance_data.get("images", [])
    if not images:
        raise ValueError("No image records were found in the instances file.")

    image_root = Path(args.images).resolve()
    accepted = []
    excluded = []
    hashes = {}
    for image in sorted(images, key=lambda item: str(item["id"])):
        source = image_root / image["file_name"]
        if not source.is_file():
            excluded.append(
                {"image_id": image["id"], "file_name": image["file_name"], "reason": "missing"}
            )
            continue
        try:
            validate_image(source)
        except (OSError, ValueError, SyntaxError) as error:
            excluded.append(
                {
                    "image_id": image["id"],
                    "file_name": image["file_name"],
                    "reason": "corrupt",
                    "detail": str(error),
                }
            )
            continue

        digest = sha256_file(source)
        if digest in hashes:
            excluded.append(
                {
                    "image_id": image["id"],
                    "file_name": image["file_name"],
                    "reason": "duplicate",
                    "duplicate_of": hashes[digest],
                }
            )
            continue
        hashes[digest] = image["id"]
        accepted.append(
            {
                "image_id": image["id"],
                "file_name": image["file_name"],
                "path": str(source),
                "sha256": digest,
            }
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(
        output / "clean_manifest.csv",
        accepted,
        ["image_id", "file_name", "path", "sha256"],
    )
    counts = Counter(item["reason"] for item in excluded)
    write_json(
        output / "cleaning_report.json",
        {
            "source_image_count": len(images),
            "clean_image_count": len(accepted),
            "excluded_image_count": len(excluded),
            "excluded_by_reason": dict(sorted(counts.items())),
            "excluded_images": excluded,
        },
    )

    if args.copy_to:
        destination = Path(args.copy_to).resolve()
        if destination == image_root:
            raise ValueError("--copy-to must not be the source image directory.")
        targets = [destination / row["file_name"] for row in accepted]
        if len(set(targets)) != len(targets):
            raise ValueError("Image filenames collide in the clean output directory.")
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite existing cleaned images: {existing[0]}"
            )
        destination.mkdir(parents=True, exist_ok=True)
        for row, target in zip(accepted, targets):
            shutil.copy2(row["path"], target)

    print(
        f"Kept {len(accepted)} of {len(images)} images; "
        f"cleaning manifest written to {(output / 'clean_manifest.csv').resolve()}"
    )


def read_manifest(path):
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"image_id", "file_name", "path", "sha256"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain columns: {', '.join(sorted(required))}")
    if not rows:
        raise ValueError(f"Manifest contains no image records: {path}")
    return rows


def select_splits(records, validation_count, seed):
    if validation_count <= 0 or validation_count >= len(records):
        raise ValueError("Validation count must be between 1 and image_count - 1.")
    validation_ids = {
        row["image_id"]
        for row in random.Random(seed).sample(records, validation_count)
    }
    validation = [row for row in records if row["image_id"] in validation_ids]
    training = [row for row in records if row["image_id"] not in validation_ids]
    return training, validation


def enumerate_test_images(directory):
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Test image validation requires Pillow. Install requirements.txt."
        ) from error

    rows = []
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError, SyntaxError) as error:
            raise ValueError(f"Invalid test image {path}: {error}") from error
        rows.append(
            {
                "image_id": path.stem,
                "file_name": path.name,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )
    return rows


def split(args):
    records = read_manifest(args.manifest)
    if len(records) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} clean knowledge-base images, "
            f"found {len(records)}."
        )
    if len({row["image_id"] for row in records}) != len(records):
        raise ValueError("Clean manifest contains duplicate image IDs.")

    test_records = enumerate_test_images(args.test_images)
    if len(test_records) != args.test_count:
        raise ValueError(
            f"Expected exactly {args.test_count} valid external test images, "
            f"found {len(test_records)}."
        )
    train, validation = select_splits(records, args.validation_count, args.seed)
    train_hashes = {row["sha256"] for row in records}
    overlap = train_hashes.intersection(row["sha256"] for row in test_records)
    if overlap:
        raise ValueError(
            f"{len(overlap)} test image(s) duplicate content in the knowledge base."
        )

    output = Path(args.output)
    write_csv(output / "knowledge_base.csv", records, list(records[0]))
    write_csv(output / "train.csv", train, list(records[0]))
    write_csv(output / "validation.csv", validation, list(records[0]))
    write_csv(output / "test.csv", test_records, list(records[0]))
    write_json(
        output / "splits.json",
        {
            "seed": args.seed,
            "knowledge_base_count": len(records),
            "training_count": len(train),
            "validation_count": len(validation),
            "test_count": len(test_records),
            "validation_is_subset_of_knowledge_base": True,
            "test_is_content_disjoint_from_knowledge_base": True,
        },
    )
    print(f"Split manifests written to {output.resolve()}")


def embed(args):
    try:
        import numpy as np
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as error:
        raise RuntimeError(
            "Embedding generation requires numpy, Pillow, torch, and transformers. "
            "Install requirements-embeddings.txt."
        ) from error

    records = read_manifest(args.manifest)
    output = Path(args.output)
    if output.suffix.lower() != ".npz":
        raise ValueError("Embedding output path must use the .npz extension.")
    captions_by_image = defaultdict(list)
    if args.captions:
        for annotation in load_json(args.captions).get("annotations", []):
            captions_by_image[str(annotation["image_id"])].append(
                annotation["caption"]
            )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(device)
    model.eval()
    vectors = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        images = []
        for row in batch:
            with Image.open(row["path"]) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
        vectors.append(features.cpu().numpy())
        for image in images:
            image.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.concatenate(vectors, axis=0)
    np.savez_compressed(
        output,
        image_ids=np.asarray([row["image_id"] for row in records]),
        embeddings=embeddings,
    )
    write_json(
        output.with_suffix(".json"),
        {
            "model": args.model,
            "count": len(records),
            "dimension": int(embeddings.shape[1]),
            "normalized": True,
            "manifest": str(Path(args.manifest).resolve()),
            "images": [
                {
                    "image_id": row["image_id"],
                    "file_name": row["file_name"],
                    "path": row["path"],
                    "captions": captions_by_image[row["image_id"]],
                }
                for row in records
            ],
        },
    )
    print(f"Saved {len(records)} image embeddings to {output.resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    explore_parser = commands.add_parser("explore", help="Create statistics and plots.")
    explore_parser.add_argument("--instances", required=True)
    explore_parser.add_argument("--captions", required=True)
    explore_parser.add_argument("--output", required=True)
    explore_parser.set_defaults(func=explore)

    clean_parser = commands.add_parser("clean", help="Detect invalid and duplicate images.")
    clean_parser.add_argument("--images", required=True, help="Directory containing COCO images.")
    clean_parser.add_argument("--instances", required=True)
    clean_parser.add_argument("--output", required=True, help="Directory for manifest and report.")
    clean_parser.add_argument("--copy-to", help="Optional directory for physical copies of kept images.")
    clean_parser.set_defaults(func=clean)

    split_parser = commands.add_parser("split", help="Create reproducible train/validation/test manifests.")
    split_parser.add_argument("--manifest", required=True, help="Input clean_manifest.csv.")
    split_parser.add_argument("--test-images", required=True, help="Separate external test image directory.")
    split_parser.add_argument("--output", required=True)
    split_parser.add_argument("--expected-count", type=int, default=10000)
    split_parser.add_argument("--validation-count", type=int, default=100)
    split_parser.add_argument("--test-count", type=int, default=100)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.set_defaults(func=split)

    embed_parser = commands.add_parser("embed", help="Build normalized CLIP image embeddings.")
    embed_parser.add_argument("--manifest", required=True, help="Use train.csv to avoid validation leakage.")
    embed_parser.add_argument("--captions", help="Optional COCO captions JSON to attach as metadata.")
    embed_parser.add_argument("--output", required=True, help="Output .npz file.")
    embed_parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    embed_parser.add_argument("--batch-size", type=int, default=32)
    embed_parser.add_argument("--device", help="Defaults to CUDA when available, otherwise CPU.")
    embed_parser.set_defaults(func=embed)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "batch_size", 1) <= 0:
        parser.error("--batch-size must be greater than zero.")
    try:
        args.func(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
