#!/usr/bin/env python3
"""Explore and prepare an MS-COCO image-captioning dataset."""

import argparse
import csv
import hashlib
import itertools
import json
import random
import re
import shutil
import sys
import time
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


def clean_caption(caption):
    """Normalize captions for retrieval while leaving source text untouched."""
    return " ".join(caption.lower().split())


def image_id_key(image_id):
    value = str(image_id)
    return str(int(value)) if value.isdecimal() else value


def read_coco_captions(path):
    captions = load_json(path).get("annotations", [])
    for annotation in captions:
        if (
            not isinstance(annotation.get("caption"), str)
            or not annotation["caption"].strip()
        ):
            raise ValueError(
                f"Caption annotation {annotation.get('id')} has no usable text."
            )
    return captions


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
    categories_by_image = defaultdict(set)
    relation_counts = Counter()
    attribute_counts = Counter()
    context_counts = Counter()
    for annotation in instances:
        image_id = image_id_key(annotation["image_id"])
        categories_by_image[image_id].add(
            categories.get(
                annotation.get("category_id"),
                str(annotation.get("category_id", "")),
            )
        )
        relations = annotation.get(
            "relationships", annotation.get("relations", [])
        )
        if not isinstance(relations, list):
            relations = [relations]
        relation_counts.update(str(value) for value in relations if value)
        attributes = annotation.get(
            "attributes", annotation.get("attribute_ids", [])
        )
        if not isinstance(attributes, list):
            attributes = [attributes]
        attribute_counts.update(str(value) for value in attributes if value)
    for image in images:
        for key in ("scene", "context", "location", "weather", "timeofday"):
            value = image.get(key)
            if value:
                context_counts[f"{key}:{value}"] += 1

    rows = image_statistics(images, instances, captions)
    caption_lengths = [
        len(annotation["caption"].split()) for annotation in captions
    ]
    tokens = [
        token.lower()
        for annotation in captions
        for token in re.findall(r"\b[\w'-]+\b", annotation["caption"])
    ]
    object_counts = [row["object_count"] for row in rows]
    caption_counts = [row["caption_count"] for row in rows]
    cooccurrence = Counter()
    for image_categories in categories_by_image.values():
        cooccurrence.update(itertools.combinations(sorted(image_categories), 2))
    spatial_relations = Counter()
    for annotation in instances:
        for key in ("relations", "relationships", "spatial_relations"):
            values = annotation.get(key, [])
            if not isinstance(values, list):
                values = [values]
            spatial_relations.update(str(value) for value in values if value)
    pos_counts = Counter()
    pos_status = "not_requested"
    if args.spacy_model:
        try:
            import spacy
        except ImportError as error:
            raise RuntimeError(
                "POS analysis requires spaCy. Install requirements-analysis.txt."
            ) from error
        nlp = spacy.load(args.spacy_model)
        pos_counts.update(
            token.pos_
            for doc in nlp.pipe(
                (annotation["caption"] for annotation in captions),
                batch_size=128,
            )
            for token in doc
            if not token.is_space
        )
        pos_status = f"completed:{args.spacy_model}"

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
        "vocabulary_size": len(set(tokens)),
        "pos_analysis": pos_status,
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
        output / "top_categories.csv",
        [
            {"category": category, "object_count": count}
            for category, count in category_counts.most_common(50)
        ],
        ["category", "object_count"],
    )
    write_csv(
        output / "object_cooccurrence.csv",
        [
            {"category_a": pair[0], "category_b": pair[1], "image_count": count}
            for pair, count in cooccurrence.most_common(500)
        ],
        ["category_a", "category_b", "image_count"],
    )
    write_csv(
        output / "spatial_relations.csv",
        [
            {"relation": relation, "count": count}
            for relation, count in spatial_relations.most_common()
        ],
        ["relation", "count"],
    )
    write_csv(
        output / "attributes.csv",
        [
            {"attribute": attribute, "count": count}
            for attribute, count in attribute_counts.most_common()
        ],
        ["attribute", "count"],
    )
    write_csv(
        output / "context_diversity.csv",
        [
            {"context": context, "count": count}
            for context, count in context_counts.most_common()
        ],
        ["context", "count"],
    )
    write_csv(
        output / "caption_vocabulary.csv",
        [
            {"token": token, "count": count}
            for token, count in Counter(tokens).most_common()
        ],
        ["token", "count"],
    )
    write_csv(
        output / "pos_distribution.csv",
        [
            {"part_of_speech": part_of_speech, "count": count}
            for part_of_speech, count in sorted(pos_counts.items())
        ],
        ["part_of_speech", "count"],
    )
    write_json(
        output / "text_statistics.json",
        {
            "caption_count": len(captions),
            "vocabulary_size": len(set(tokens)),
            "mean_caption_words": summary["mean_caption_words"],
            "pos_analysis": pos_status,
        },
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
    caption_audit = None
    if args.captions:
        image_ids = {image_id_key(image["id"]) for image in images}
        caption_rows_result, mismatches, distribution = caption_rows(
            read_coco_captions(args.captions),
            image_ids,
            args.captions_per_image,
        )
        caption_audit = {
            "expected_captions_per_image": args.captions_per_image,
            "caption_count": len(caption_rows_result),
            "caption_count_distribution": distribution,
            "images_with_unexpected_caption_count": mismatches,
        }
    write_json(
        output / "cleaning_report.json",
        {
            "source_image_count": len(images),
            "clean_image_count": len(accepted),
            "excluded_image_count": len(excluded),
            "excluded_by_reason": dict(sorted(counts.items())),
            "excluded_images": excluded,
            "caption_integrity": caption_audit,
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
    for row in rows:
        missing_values = [
            field
            for field in required
            if not str(row.get(field) or "").strip()
        ]
        if missing_values:
            raise ValueError(
                f"Manifest {path} has empty required values: "
                f"{', '.join(sorted(missing_values))}."
            )
        row["image_id"] = image_id_key(row["image_id"])
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
                "image_id": image_id_key(path.stem),
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
    test_ids = [row["image_id"] for row in test_records]
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("External test images contain duplicate image IDs.")
    if set(test_ids).intersection(row["image_id"] for row in records):
        raise ValueError("External test image IDs overlap with the knowledge base.")
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
            "version": 1,
            "seed": args.seed,
            "knowledge_base_count": len(records),
            "training_count": len(train),
            "validation_count": len(validation),
            "test_count": len(test_records),
            "validation_is_subset_of_knowledge_base": True,
            "test_is_content_disjoint_from_knowledge_base": True,
            "test_reference_captions_in_knowledge_base": False,
            "test_self_retrieval_blocked": True,
        },
    )
    write_json(
        output / "split_ids.json",
        {
            "version": 1,
            "seed": args.seed,
            "knowledge_base": [row["image_id"] for row in records],
            "train": [row["image_id"] for row in train],
            "validation": [row["image_id"] for row in validation],
            "test": [row["image_id"] for row in test_records],
        },
    )
    print(f"Split manifests written to {output.resolve()}")


def caption_rows(captions, image_ids, expected_count):
    rows_by_image = defaultdict(list)
    for annotation in captions:
        image_id = image_id_key(annotation["image_id"])
        if image_id in image_ids:
            text = annotation["caption"]
            rows_by_image[image_id].append(
                {
                    "caption_id": annotation.get("id", len(rows_by_image[image_id])),
                    "image_id": image_id,
                    "raw_caption": text,
                    "cleaned_caption": clean_caption(text),
                }
            )

    result = [
        row
        for image_id in sorted(image_ids)
        for row in rows_by_image[image_id]
    ]
    counts = Counter(
        len(rows_by_image[image_id]) for image_id in image_ids
    )
    mismatches = [
        {"image_id": image_id, "caption_count": len(rows_by_image[image_id])}
        for image_id in sorted(image_ids)
        if expected_count and len(rows_by_image[image_id]) != expected_count
    ]
    return result, mismatches, dict(sorted(counts.items()))


def image_metadata(image_records, instance_data):
    categories = {
        category["id"]: category["name"]
        for category in instance_data.get("categories", [])
    }
    annotations_by_image = defaultdict(list)
    for annotation in instance_data.get("annotations", []):
        annotations_by_image[image_id_key(annotation["image_id"])].append(
            annotation
        )

    metadata = []
    for image in image_records:
        image_id = image_id_key(image["image_id"])
        annotations = annotations_by_image[image_id]
        objects = [
            {
                "category": categories.get(
                    annotation.get("category_id"),
                    str(annotation.get("category_id", "")),
                ),
                "bbox": annotation.get("bbox"),
                "area": annotation.get("area"),
                "confidence": annotation.get("score"),
                "attributes": annotation.get(
                    "attributes", annotation.get("attribute_ids", [])
                ),
                "relationships": annotation.get(
                    "relationships", annotation.get("relations", [])
                ),
            }
            for annotation in annotations
        ]
        image_fields = {
            key: image.get(key)
            for key in (
                "scene",
                "context",
                "location",
                "weather",
                "timeofday",
                "split",
            )
            if image.get(key) is not None
        }
        metadata.append(
            {
                "image_id": image_id,
                "objects": objects,
                "attributes": [
                    obj["attributes"] for obj in objects if obj["attributes"]
                ],
                "relationships": [
                    obj["relationships"]
                    for obj in objects
                    if obj["relationships"]
                ],
                "context": image_fields,
            }
        )
    return metadata


def preprocess(args):
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise RuntimeError(
            "Image preprocessing requires Pillow. Install requirements.txt."
        ) from error

    split_root = Path(args.splits)
    manifests = {
        name: read_manifest(split_root / f"{name}.csv")
        for name in ("knowledge_base", "train", "validation", "test")
    }
    instance_data = load_json(args.instances)
    if args.test_instances:
        test_instance_data = load_json(args.test_instances)
        category_names = {
            category["id"]: category["name"]
            for category in instance_data.get("categories", [])
        }
        for category in test_instance_data.get("categories", []):
            existing_name = category_names.get(category["id"])
            if existing_name and existing_name != category["name"]:
                raise ValueError(
                    f"Conflicting category names for ID {category['id']}."
                )
            category_names[category["id"]] = category["name"]
        instance_data["categories"] = [
            {"id": category_id, "name": name}
            for category_id, name in sorted(category_names.items())
        ]
        instance_data["images"] = (
            instance_data.get("images", [])
            + test_instance_data.get("images", [])
        )
        instance_data["annotations"] = (
            instance_data.get("annotations", [])
            + test_instance_data.get("annotations", [])
        )
    images_by_id = {
        image_id_key(image["id"]): image
        for image in instance_data.get("images", [])
    }
    if len(images_by_id) != len(instance_data.get("images", [])):
        raise ValueError("Instances annotations contain duplicate image IDs.")
    known_ids = set(images_by_id)
    knowledge_base_ids = {
        row["image_id"] for row in manifests["knowledge_base"]
    }
    if not knowledge_base_ids.issubset(known_ids):
        raise ValueError(
            "The instances annotations do not contain every knowledge-base image."
        )
    train_ids = {row["image_id"] for row in manifests["train"]}
    validation_ids = {row["image_id"] for row in manifests["validation"]}
    test_ids = {row["image_id"] for row in manifests["test"]}
    if train_ids & validation_ids:
        raise ValueError("Train and validation image IDs overlap.")
    if train_ids | validation_ids != knowledge_base_ids:
        raise ValueError("Train and validation do not cover the knowledge base.")
    if not validation_ids.issubset(knowledge_base_ids):
        raise ValueError("Validation IDs are outside the knowledge base.")
    if (train_ids | validation_ids) & test_ids:
        raise ValueError("Test image IDs overlap with train or validation IDs.")
    knowledge_base_hashes = {
        row["sha256"] for row in manifests["knowledge_base"]
    }
    test_hashes = {row["sha256"] for row in manifests["test"]}
    if knowledge_base_hashes.intersection(test_hashes):
        raise ValueError("Test image content overlaps with the knowledge base.")

    output = Path(args.output).resolve()
    image_output = output / "images"
    image_output.mkdir(parents=True, exist_ok=True)
    processed_by_id = {}
    seen_targets = set()
    for name in ("knowledge_base", "test"):
        processed_records = []
        for row in manifests[name]:
            image_id = image_id_key(row["image_id"])
            source = Path(row["path"])
            if not source.is_file():
                raise FileNotFoundError(f"Image file does not exist: {source}")
            safe_id = "".join(
                char for char in image_id if char.isalnum() or char in "-_"
            )
            if not safe_id:
                raise ValueError(f"Image ID cannot form a safe filename: {image_id}")
            target = image_output / f"{safe_id}_{Path(row['file_name']).stem}.jpg"
            if target in seen_targets:
                raise ValueError(f"Processed image filename collision: {target}")
            seen_targets.add(target)
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                if args.max_size > 0 and max(image.size) > args.max_size:
                    image.thumbnail(
                        (args.max_size, args.max_size),
                        Image.Resampling.LANCZOS,
                    )
                image.save(target, format="JPEG", quality=95)
                image.close()
            processed = dict(row)
            processed["path"] = str(target.resolve())
            processed["sha256"] = sha256_file(target)
            processed_by_id[image_id] = processed
            processed_records.append(processed)

    for name, records in manifests.items():
        converted = [
            processed_by_id[row["image_id"]]
            for row in records
            if row["image_id"] in processed_by_id
        ]
        if len(converted) != len(records):
            raise ValueError(f"Could not preprocess every {name} image.")
        write_csv(output / f"{name}.csv", converted, list(records[0]))

    captions = read_coco_captions(args.captions)
    train_caption_rows, train_mismatches, train_distribution = caption_rows(
        captions, train_ids, args.captions_per_image
    )
    write_csv(
        output / "train_captions.csv",
        train_caption_rows,
        ["caption_id", "image_id", "raw_caption", "cleaned_caption"],
    )
    caption_reports = {
        "train": {
            "caption_count": len(train_caption_rows),
            "images_with_unexpected_caption_count": train_mismatches,
            "caption_count_distribution": train_distribution,
        }
    }
    for name, image_ids, path in (
        ("validation", validation_ids, args.validation_captions),
        ("test", test_ids, args.test_captions),
    ):
        if not path:
            continue
        split_captions, mismatches, distribution = caption_rows(
            read_coco_captions(path), image_ids, args.captions_per_image
        )
        write_csv(
            output / f"{name}_captions.csv",
            split_captions,
            ["caption_id", "image_id", "raw_caption", "cleaned_caption"],
        )
        caption_reports[name] = {
            "caption_count": len(split_captions),
            "images_with_unexpected_caption_count": mismatches,
            "caption_count_distribution": distribution,
        }

    manifest_rows_by_id = {
        row["image_id"]: row
        for records in manifests.values()
        for row in records
    }
    annotation_images = [
        images_by_id.get(
            image_id,
            {
                "id": image_id,
                "file_name": manifest_rows_by_id[image_id]["file_name"],
            },
        )
        for image_id in sorted(manifest_rows_by_id)
    ]
    metadata = image_metadata(
        [
            {
                "image_id": image["id"],
                "file_name": image["file_name"],
            }
            for image in annotation_images
        ],
        instance_data,
    )
    write_json(output / "metadata.json", metadata)
    write_csv(
        output / "metadata.csv",
        [
            {
                "image_id": row["image_id"],
                "objects": json.dumps(row["objects"], ensure_ascii=False),
                "attributes": json.dumps(row["attributes"], ensure_ascii=False),
                "relationships": json.dumps(
                    row["relationships"], ensure_ascii=False
                ),
                "context": json.dumps(row["context"], ensure_ascii=False),
            }
            for row in metadata
        ],
        ["image_id", "objects", "attributes", "relationships", "context"],
    )
    write_json(
        output / "preprocessing_report.json",
        {
            "version": 1,
            "images_preprocessed": len(processed_by_id),
            "image_format": "JPEG RGB, EXIF orientation applied",
            "max_size": args.max_size or None,
            "aspect_ratio_preserved": True,
            "encoder_resize_normalization": (
                "Applied by the selected encoder processor during embedding."
            ),
            "captions_per_image_expected": args.captions_per_image,
            "caption_audit": caption_reports,
            "detector": {
                "name": args.detector_name,
                "version": args.detector_version,
                "confidence_threshold": args.confidence_threshold,
                "executed": False,
            },
        },
    )
    print(f"Preprocessed dataset artifacts written to {output}")


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
    record_ids = {row["image_id"] for row in records}
    caption_records = []
    if args.captions:
        caption_path = Path(args.captions)
        if caption_path.suffix.lower() == ".csv":
            with caption_path.open("r", encoding="utf-8", newline="") as file:
                caption_records = list(csv.DictReader(file))
            for index, row in enumerate(caption_records):
                if not str(row.get("image_id") or "").strip():
                    raise ValueError("Caption CSV contains a row without an image_id.")
                row.setdefault("caption_id", str(index))
                row["image_id"] = image_id_key(row["image_id"])
                row["raw_caption"] = str(
                    row.get("raw_caption") or row.get("caption") or ""
                )
                if not row["raw_caption"].strip():
                    raise ValueError(
                        f"Caption CSV row {index + 2} contains no usable text."
                    )
                row["cleaned_caption"] = row.get(
                    "cleaned_caption"
                ) or clean_caption(row["raw_caption"])
        else:
            for annotation in read_coco_captions(caption_path):
                caption_records.append(
                    {
                        "caption_id": annotation.get(
                            "id", len(caption_records)
                        ),
                        "image_id": image_id_key(annotation["image_id"]),
                        "raw_caption": annotation["caption"],
                        "cleaned_caption": clean_caption(
                            annotation["caption"]
                        ),
                    }
                )
        outside_ids = {
            row["image_id"] for row in caption_records
        } - record_ids
        if outside_ids:
            raise ValueError(
                "Caption input includes image IDs outside the embedding "
                f"manifest (possible split leakage): {sorted(outside_ids)[:5]}"
            )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(device)
    model.eval()
    image_vectors = []
    text_vectors = []
    started = time.perf_counter()
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
        image_vectors.append(features.cpu().numpy())
        for image in images:
            image.close()

    for start in range(0, len(caption_records), args.batch_size):
        batch = caption_records[start : start + args.batch_size]
        texts = [row["cleaned_caption"] for row in batch]
        inputs = processor(
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
        text_vectors.append(features.cpu().numpy())

    output.parent.mkdir(parents=True, exist_ok=True)
    image_embeddings = np.concatenate(image_vectors, axis=0).astype(
        np.float32
    )
    dimension = int(image_embeddings.shape[1])
    text_embeddings = (
        np.concatenate(text_vectors, axis=0).astype(np.float32)
        if text_vectors
        else np.empty((0, dimension), dtype=np.float32)
    )
    np.savez_compressed(
        output,
        image_ids=np.asarray([row["image_id"] for row in records]),
        embeddings=image_embeddings,
        image_embeddings=image_embeddings,
        caption_image_ids=np.asarray(
            [row["image_id"] for row in caption_records]
        ),
        caption_ids=np.asarray(
            [str(row["caption_id"]) for row in caption_records]
        ),
        captions=np.asarray(
            [row["raw_caption"] for row in caption_records]
        ),
        cleaned_captions=np.asarray(
            [row["cleaned_caption"] for row in caption_records]
        ),
        text_embeddings=text_embeddings,
    )
    write_json(
        output.with_suffix(".json"),
        {
            "model": args.model,
            "count": len(records),
            "image_embedding_count": len(image_embeddings),
            "text_embedding_count": len(text_embeddings),
            "dimension": dimension,
            "normalized": True,
            "similarity_metric": "cosine",
            "batch_size": args.batch_size,
            "device": device,
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "manifest": str(Path(args.manifest).resolve()),
            "images": [
                {
                    "image_id": row["image_id"],
                    "file_name": row["file_name"],
                    "path": row["path"],
                }
                for row in records
            ],
        },
    )
    print(
        f"Saved {len(image_embeddings)} image and {len(text_embeddings)} "
        f"text embeddings to {output.resolve()}"
    )


def load_embeddings(path):
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Embedding inspection requires numpy. Install requirements-embeddings.txt."
        ) from error

    with np.load(path, allow_pickle=False) as data:
        embedding_key = (
            "image_embeddings" if "image_embeddings" in data else "embeddings"
        )
        if "image_ids" not in data or embedding_key not in data:
            raise ValueError(
                "Embedding archive must contain image_ids and image_embeddings."
            )
        image_embeddings = data[
            embedding_key
        ].astype(np.float32)
        if image_embeddings.ndim != 2:
            raise ValueError("Image embeddings must be a 2D matrix.")
        image_ids = data["image_ids"].astype(str)
        text_embeddings = (
            data["text_embeddings"].astype(np.float32)
            if "text_embeddings" in data
            else np.empty((0, image_embeddings.shape[1]), dtype=np.float32)
        )
        if text_embeddings.ndim != 2:
            raise ValueError("Text embeddings must be a 2D matrix.")
        caption_image_ids = (
            data["caption_image_ids"].astype(str)
            if "caption_image_ids" in data
            else np.asarray([], dtype=str)
        )
        caption_ids = (
            data["caption_ids"].astype(str)
            if "caption_ids" in data
            else np.asarray([], dtype=str)
        )
        captions = (
            data["captions"].astype(str)
            if "captions" in data
            else np.asarray([], dtype=str)
        )
        cleaned_captions = (
            data["cleaned_captions"].astype(str)
            if "cleaned_captions" in data
            else captions
        )
    return {
        "image_ids": image_ids,
        "image_embeddings": image_embeddings,
        "caption_image_ids": caption_image_ids,
        "caption_ids": caption_ids,
        "captions": captions,
        "cleaned_captions": cleaned_captions,
        "text_embeddings": text_embeddings,
    }


def build_index(args):
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Index construction requires numpy. Install requirements-embeddings.txt."
        ) from error

    started = time.perf_counter()
    data = load_embeddings(args.embeddings)
    if data["image_embeddings"].ndim != 2 or not len(data["image_ids"]):
        raise ValueError("Image embeddings must be a non-empty 2D matrix.")
    if len(set(data["image_ids"])) != len(data["image_ids"]):
        raise ValueError("Image embeddings contain duplicate image IDs.")
    if data["text_embeddings"].ndim != 2:
        raise ValueError("Text embeddings must be a 2D matrix.")
    if data["text_embeddings"].shape[1] != data["image_embeddings"].shape[1]:
        raise ValueError("Image and text embedding dimensions do not match.")
    if len(data["text_embeddings"]) != len(data["caption_image_ids"]):
        raise ValueError("Text embeddings and caption metadata counts differ.")
    if len(data["caption_ids"]) != len(data["text_embeddings"]):
        raise ValueError("Text embeddings and caption IDs counts differ.")
    if len(data["captions"]) != len(data["text_embeddings"]):
        raise ValueError("Text embeddings and captions counts differ.")
    if not set(data["caption_image_ids"]).issubset(set(data["image_ids"])):
        raise ValueError("A caption embedding is not linked to an indexed image.")
    caption_vector_ids = [
        f"{image_id}:{caption_id}"
        for image_id, caption_id in zip(
            data["caption_image_ids"], data["caption_ids"]
        )
    ]
    if len(caption_vector_ids) != len(set(caption_vector_ids)):
        raise ValueError("Caption embeddings contain duplicate caption IDs.")
    if not np.isfinite(data["image_embeddings"]).all():
        raise ValueError("Image embeddings contain non-finite values.")
    if not np.isfinite(data["text_embeddings"]).all():
        raise ValueError("Text embeddings contain non-finite values.")

    metadata_rows = load_json(args.metadata)
    if not isinstance(metadata_rows, list):
        raise ValueError("Metadata JSON must contain a list of image records.")
    if any(
        not isinstance(row, dict) or "image_id" not in row
        for row in metadata_rows
    ):
        raise ValueError("Every metadata record must include an image_id.")
    if any("image_id" not in row for row in metadata_rows):
        raise ValueError("Every metadata record must include an image_id.")
    metadata_by_id = {image_id_key(row["image_id"]): row for row in metadata_rows}
    if len(metadata_by_id) != len(metadata_rows):
        raise ValueError("Metadata JSON contains duplicate image IDs.")
    missing_metadata = sorted(
        set(data["image_ids"].tolist())
        .union(data["caption_image_ids"].tolist())
        .difference(metadata_by_id)
    )
    if missing_metadata:
        raise ValueError(
            "Missing metadata for indexed image IDs: "
            f"{missing_metadata[:5]}"
        )

    vectors = [data["image_embeddings"]]
    vector_ids = [f"image:{image_id}" for image_id in data["image_ids"]]
    vector_image_ids = list(data["image_ids"])
    modalities = ["image"] * len(data["image_ids"])
    captions = [""] * len(data["image_ids"])
    vector_metadata = [
        json.dumps(metadata_by_id[image_id], ensure_ascii=False)
        for image_id in data["image_ids"]
    ]
    vectors.append(data["text_embeddings"])
    vector_ids.extend(
        f"caption:{image_id}:{caption_id}"
        for image_id, caption_id in zip(
            data["caption_image_ids"], data["caption_ids"]
        )
    )
    vector_image_ids.extend(data["caption_image_ids"])
    modalities.extend(["text"] * len(data["text_embeddings"]))
    captions.extend(data["captions"])
    vector_metadata.extend(
        json.dumps(metadata_by_id[image_id], ensure_ascii=False)
        for image_id in data["caption_image_ids"]
    )
    if len(vector_ids) != len(set(vector_ids)):
        raise ValueError("Index contains duplicate vector IDs.")
    index_vectors = np.concatenate(vectors, axis=0).astype(np.float32)
    norms = np.linalg.norm(index_vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot build a cosine index with zero-length vectors.")
    index_vectors /= norms
    output = Path(args.output)
    if output.suffix.lower() != ".npz":
        raise ValueError("Index output path must use the .npz extension.")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        vectors=index_vectors,
        vector_ids=np.asarray(vector_ids),
        image_ids=np.asarray(vector_image_ids),
        modalities=np.asarray(modalities),
        captions=np.asarray(captions),
        metadata=np.asarray(vector_metadata),
    )
    write_json(
        output.with_suffix(".json"),
        {
            "version": 1,
            "index_type": "exact flat cosine",
            "similarity_metric": "cosine",
            "dimension": int(index_vectors.shape[1]),
            "vector_count": len(index_vectors),
            "image_vector_count": len(data["image_ids"]),
            "text_vector_count": len(data["text_embeddings"]),
            "metadata_linked": True,
            "metadata_source": str(Path(args.metadata).resolve()),
            "source_embeddings": str(Path(args.embeddings).resolve()),
            "build_seconds": round(time.perf_counter() - started, 3),
        },
    )
    print(f"Built exact cosine index with {len(index_vectors)} vectors at {output}")


def quality_check(args):
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Quality checks require numpy. Install requirements-embeddings.txt."
        ) from error

    records = read_manifest(args.manifest)
    data = load_embeddings(args.embeddings)
    manifest_ids = [row["image_id"] for row in records]
    embedding_ids = data["image_ids"].tolist()
    issues = []
    if len(manifest_ids) != len(set(manifest_ids)):
        issues.append("Manifest contains duplicate image IDs.")
    if len(embedding_ids) != len(set(embedding_ids)):
        issues.append("Image embeddings contain duplicate image IDs.")
    if embedding_ids != manifest_ids:
        issues.append("Image embedding IDs or ordering differ from the manifest.")
    missing_embeddings = sorted(set(manifest_ids) - set(embedding_ids))
    unexpected_embeddings = sorted(set(embedding_ids) - set(manifest_ids))
    if missing_embeddings:
        issues.append(f"Missing image embeddings for {len(missing_embeddings)} IDs.")
    if unexpected_embeddings:
        issues.append(f"Embeddings include {len(unexpected_embeddings)} unexpected IDs.")
    if len(data["image_embeddings"]) != len(data["image_ids"]):
        issues.append("Image embedding count does not match image ID count.")
    if (
        len(data["text_embeddings"]) != len(data["caption_image_ids"])
        or len(data["text_embeddings"]) != len(data["caption_ids"])
        or len(data["text_embeddings"]) != len(data["captions"])
        or len(data["text_embeddings"]) != len(data["cleaned_captions"])
    ):
        issues.append("Text embedding counts do not match caption metadata.")
    if data["image_embeddings"].ndim != 2:
        issues.append("Image embeddings are not a 2D matrix.")
    if data["text_embeddings"].ndim != 2:
        issues.append("Text embeddings are not a 2D matrix.")
    if (
        data["image_embeddings"].ndim == 2
        and data["text_embeddings"].ndim == 2
        and data["image_embeddings"].shape[1]
        != data["text_embeddings"].shape[1]
    ):
        issues.append("Image and text embedding dimensions do not match.")
    if not np.isfinite(data["image_embeddings"]).all():
        issues.append("Image embeddings contain NaN or infinite values.")
    if not np.isfinite(data["text_embeddings"]).all():
        issues.append("Text embeddings contain NaN or infinite values.")
    image_norms = (
        np.linalg.norm(data["image_embeddings"], axis=1)
        if data["image_embeddings"].ndim == 2
        else np.asarray([])
    )
    if len(image_norms) and not np.allclose(image_norms, 1, atol=1e-3):
        issues.append("Image embeddings are not L2-normalized.")
    text_norms = (
        np.linalg.norm(data["text_embeddings"], axis=1)
        if data["text_embeddings"].ndim == 2
        else np.asarray([])
    )
    if len(text_norms) and not np.allclose(text_norms, 1, atol=1e-3):
        issues.append("Text embeddings are not L2-normalized.")

    caption_count_mismatch = None
    caption_reference_count_mismatch = None
    if args.captions:
        with Path(args.captions).open(
            "r", encoding="utf-8", newline=""
        ) as file:
            caption_rows_input = list(csv.DictReader(file))
        expected = [
            (
                row["image_id"],
                str(row["caption_id"]),
                row.get("raw_caption", row.get("caption", "")),
                row.get("cleaned_caption", clean_caption(row.get("raw_caption", row.get("caption", "")))),
            )
            for row in caption_rows_input
        ]
        actual = list(
            zip(
                data["caption_image_ids"].tolist(),
                data["caption_ids"].tolist(),
                data["captions"].tolist(),
                data["cleaned_captions"].tolist(),
            )
        )
        caption_count_mismatch = []
        if expected != actual:
            expected_counts = Counter(row[0] for row in expected)
            actual_counts = Counter(row[0] for row in actual)
            caption_count_mismatch = sorted(
                image_id
                for image_id in set(expected_counts) | set(actual_counts)
                if expected_counts[image_id] != actual_counts[image_id]
            )
            if not caption_count_mismatch:
                caption_count_mismatch = ["caption_content_or_order"]
        if caption_count_mismatch:
            issues.append(
                "Caption manifest and text embeddings differ for "
                f"{len(caption_count_mismatch)} item(s)."
            )
        if len(data["caption_image_ids"]) != len(set(
            zip(data["caption_image_ids"], data["caption_ids"])
        )):
            issues.append("Text embeddings contain duplicate caption IDs.")
        if args.captions_per_image:
            caption_counts = Counter(row["image_id"] for row in caption_rows_input)
            caption_reference_count_mismatch = [
                {
                    "image_id": image_id,
                    "caption_count": caption_counts[image_id],
                }
                for image_id in manifest_ids
                if caption_counts[image_id] != args.captions_per_image
            ]
            if caption_reference_count_mismatch:
                issues.append(
                    "Caption reference count differs from the expected "
                    f"{args.captions_per_image} for "
                    f"{len(caption_reference_count_mismatch)} image IDs."
                )

    split_audit = {}
    if args.splits:
        split_ids = load_json(Path(args.splits) / "split_ids.json")
        train_ids = set(split_ids["train"])
        validation_ids = set(split_ids["validation"])
        test_ids = set(split_ids["test"])
        knowledge_base_ids = set(split_ids["knowledge_base"])
        split_audit = {
            "train_test_disjoint": not train_ids.intersection(test_ids),
            "validation_test_disjoint": not validation_ids.intersection(test_ids),
            "train_validation_disjoint": not train_ids.intersection(
                validation_ids
            ),
            "validation_is_subset_of_knowledge_base": validation_ids.issubset(
                knowledge_base_ids
            ),
            "train_and_validation_cover_knowledge_base": (
                train_ids | validation_ids
            )
            == knowledge_base_ids,
        }
        if not all(split_audit.values()):
            issues.append("Split ID audit found overlapping image IDs.")
        if set(embedding_ids).intersection(test_ids):
            issues.append("Test images are present in retrieval embeddings.")
        if set(data["caption_image_ids"].tolist()).intersection(test_ids):
            issues.append("Test captions are present in retrieval embeddings.")

    metadata_count = None
    missing_metadata_ids = []
    if args.metadata:
        metadata_rows = load_json(args.metadata)
        if not isinstance(metadata_rows, list):
            raise ValueError("Metadata JSON must contain a list of image records.")
        metadata_ids = [
            image_id_key(row["image_id"]) for row in metadata_rows
        ]
        if len(metadata_ids) != len(set(metadata_ids)):
            issues.append("Metadata contains duplicate image IDs.")
        missing_metadata_ids = sorted(set(manifest_ids) - set(metadata_ids))
        if missing_metadata_ids:
            issues.append(
                f"Metadata is missing {len(missing_metadata_ids)} manifest IDs."
            )
        metadata_count = len(metadata_ids)

    report = {
        "version": 1,
        "manifest_count": len(manifest_ids),
        "image_embedding_count": len(embedding_ids),
        "text_embedding_count": len(data["text_embeddings"]),
        "embedding_dimension": (
            int(data["image_embeddings"].shape[1])
            if data["image_embeddings"].ndim == 2
            else None
        ),
        "missing_embedding_ids": missing_embeddings,
        "unexpected_embedding_ids": unexpected_embeddings,
        "caption_count_mismatch_ids": caption_count_mismatch,
        "caption_reference_count_mismatch": caption_reference_count_mismatch,
        "metadata_count": metadata_count,
        "missing_metadata_ids": missing_metadata_ids,
        "split_audit": split_audit,
        "issues": issues,
        "passed": not issues,
    }
    write_json(args.output, report)
    if issues:
        raise ValueError(
            f"Quality checks failed ({len(issues)} issue(s)); see {args.output}."
        )
    print(f"Quality checks passed; report written to {Path(args.output).resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    explore_parser = commands.add_parser(
        "explore", help="Create dataset statistics and visualizations."
    )
    explore_parser.add_argument("--instances", required=True)
    explore_parser.add_argument("--captions", required=True)
    explore_parser.add_argument("--output", required=True)
    explore_parser.add_argument(
        "--spacy-model",
        help="Optional spaCy model name for part-of-speech statistics.",
    )
    explore_parser.set_defaults(func=explore)

    clean_parser = commands.add_parser("clean", help="Detect invalid and duplicate images.")
    clean_parser.add_argument("--images", required=True, help="Directory containing COCO images.")
    clean_parser.add_argument("--instances", required=True)
    clean_parser.add_argument("--output", required=True, help="Directory for manifest and report.")
    clean_parser.add_argument("--copy-to", help="Optional directory for physical copies of kept images.")
    clean_parser.add_argument(
        "--captions",
        help="Optional COCO captions JSON to audit reference-caption counts.",
    )
    clean_parser.add_argument("--captions-per-image", type=int, default=5)
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

    preprocess_parser = commands.add_parser(
        "preprocess",
        help="Create RGB images, isolated caption files, and linked metadata.",
    )
    preprocess_parser.add_argument("--splits", required=True)
    preprocess_parser.add_argument("--instances", required=True)
    preprocess_parser.add_argument(
        "--test-instances",
        help="Optional COCO instances JSON for held-out test-image metadata.",
    )
    preprocess_parser.add_argument("--captions", required=True)
    preprocess_parser.add_argument(
        "--validation-captions",
        help="Optional separate COCO captions JSON for validation images.",
    )
    preprocess_parser.add_argument(
        "--test-captions",
        help="Optional separate COCO captions JSON for held-out test images.",
    )
    preprocess_parser.add_argument("--output", required=True)
    preprocess_parser.add_argument(
        "--max-size",
        type=int,
        default=0,
        help="Optional longest image edge in pixels; zero keeps source size.",
    )
    preprocess_parser.add_argument("--captions-per-image", type=int, default=5)
    preprocess_parser.add_argument("--detector-name")
    preprocess_parser.add_argument("--detector-version")
    preprocess_parser.add_argument("--confidence-threshold", type=float)
    preprocess_parser.set_defaults(func=preprocess)

    embed_parser = commands.add_parser(
        "embed", help="Build normalized CLIP image and optional text embeddings."
    )
    embed_parser.add_argument("--manifest", required=True, help="Use train.csv to avoid validation leakage.")
    embed_parser.add_argument(
        "--captions",
        help="Optional COCO captions JSON or prepared caption CSV.",
    )
    embed_parser.add_argument("--output", required=True, help="Output .npz file.")
    embed_parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    embed_parser.add_argument("--batch-size", type=int, default=32)
    embed_parser.add_argument("--device", help="Defaults to CUDA when available, otherwise CPU.")
    embed_parser.set_defaults(func=embed)

    index_parser = commands.add_parser(
        "build-index", help="Build a reproducible exact cosine vector index."
    )
    index_parser.add_argument("--embeddings", required=True)
    index_parser.add_argument("--metadata", required=True)
    index_parser.add_argument("--output", required=True, help="Output .npz file.")
    index_parser.set_defaults(func=build_index)

    quality_parser = commands.add_parser(
        "quality-check", help="Audit embeddings, captions, and split leakage."
    )
    quality_parser.add_argument("--manifest", required=True)
    quality_parser.add_argument("--embeddings", required=True)
    quality_parser.add_argument(
        "--captions", help="Prepared caption CSV corresponding to the embeddings."
    )
    quality_parser.add_argument(
        "--splits", help="Split output directory containing split_ids.json."
    )
    quality_parser.add_argument(
        "--metadata", help="Prepared metadata.json to verify against the manifest."
    )
    quality_parser.add_argument(
        "--captions-per-image",
        type=int,
        default=5,
        help="Expected reference captions per manifest image; zero disables this check.",
    )
    quality_parser.add_argument("--output", required=True, help="JSON report path.")
    quality_parser.set_defaults(func=quality_check)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "batch_size", 1) <= 0:
        parser.error("--batch-size must be greater than zero.")
    for argument in ("max_size", "captions_per_image"):
        if getattr(args, argument, 0) < 0:
            parser.error(f"--{argument.replace('_', '-')} must not be negative.")
    if (
        getattr(args, "confidence_threshold", None) is not None
        and not 0 <= args.confidence_threshold <= 1
    ):
        parser.error("--confidence-threshold must be between zero and one.")
    for argument in ("expected_count", "validation_count", "test_count"):
        if getattr(args, argument, 1) <= 0:
            parser.error(f"--{argument.replace('_', '-')} must be greater than zero.")
    try:
        args.func(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
