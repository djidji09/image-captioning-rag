#!/usr/bin/env python3
"""Explore and prepare an MS-COCO image-captioning dataset."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_config(path):
    try:
        import yaml
    except ImportError:
        yaml = None
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        if yaml is None:
            try:
                config = json.load(file)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "This YAML configuration uses features that require "
                    "PyYAML. Install requirements.txt or use JSON-compatible YAML."
                ) from error
        else:
            config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    return config, config_path.parent.parent


def configured_path(root, value):
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else (root / path).resolve())


def apply_config(args):
    if not getattr(args, "config", None):
        return
    config, root = load_config(args.config)
    dataset = config.get("dataset", {})
    annotations = dataset.get("annotations", {})
    images = dataset.get("images", {})
    outputs = config.get("outputs", {})
    selection = config.get("selection", {})
    embedding = config.get("embedding", {})
    preprocessing = config.get("preprocessing", {})
    index = config.get("index", {})
    split_root = outputs.get("splits")
    preprocessed_root = outputs.get("preprocessed")

    command_paths = {
        "audit": {
            "instances": annotations.get("train_instances"),
            "captions": annotations.get("train_captions"),
            "images": images.get("train"),
            "output": outputs.get("audit"),
        },
        "select": {
            "instances": annotations.get("train_instances"),
            "captions": annotations.get("train_captions"),
            "images": images.get("train"),
            "output": outputs.get("candidates"),
        },
        "split": {
            "manifest": outputs.get("candidate_manifest"),
            "instances": annotations.get("train_instances"),
            "captions": annotations.get("train_captions"),
            "test_instances": annotations.get("val_instances"),
            "test_captions": annotations.get("val_captions"),
            "test_images": images.get("val"),
            "output": split_root,
        },
        "preprocess": {
            "splits": split_root,
            "instances": annotations.get("train_instances"),
            "test_instances": annotations.get("val_instances"),
            "captions": annotations.get("train_captions"),
            "validation_captions": annotations.get("train_captions"),
            "test_captions": annotations.get("val_captions"),
            "output": preprocessed_root,
        },
        "embed": {
            "manifest": (
                str(Path(split_root) / "knowledge_base_manifest.csv")
                if split_root
                else None
            ),
            "captions": (
                str(Path(split_root) / "knowledge_base_captions.csv")
                if split_root
                else None
            ),
            "output": outputs.get("embeddings"),
        },
        "build-index": {
            "embeddings": outputs.get("embeddings"),
            "metadata": outputs.get("preprocessed_metadata"),
            "output": outputs.get("index_output"),
        },
        "build-faiss-index": {
            "clip_embeddings": outputs.get("clip_embeddings"),
            "sbert_embeddings": outputs.get("sbert_embeddings"),
            "metadata": outputs.get("preprocessed_metadata"),
            "output": outputs.get("faiss_index_dir"),
        },
        "quality-check": {
            "manifest": (
                str(Path(split_root) / "knowledge_base_manifest.csv")
                if split_root
                else None
            ),
            "embeddings": outputs.get("embeddings"),
            "splits": split_root,
            "preprocessed": preprocessed_root,
            "metadata": outputs.get("preprocessed_metadata"),
            "captions": (
                str(Path(split_root) / "knowledge_base_captions.csv")
                if split_root
                else None
            ),
            "output": outputs.get("quality_control"),
        },
        "release-manifest": {
            "splits": str(Path(split_root) / "splits_v2.json") if split_root else None,
            "output": "release_manifest.json",
        },
    }
    for name, value in command_paths.get(args.command, {}).items():
        if getattr(args, name, None) is None and value is not None:
            setattr(args, name, configured_path(root, value))
    encoder = getattr(args, "encoder", None)
    model_value = embedding.get("clip_model")
    if args.command == "embed" and encoder == "sbert":
        model_value = embedding.get("sentence_transformer_model")
    elif args.command == "encode-query" and encoder == "sbert-text":
        model_value = embedding.get("sentence_transformer_model")

    for name, value in {
        "candidate_count": selection.get("candidate_count"),
        "knowledge_base_count": selection.get("knowledge_base_count"),
        "validation_count": selection.get("validation_count"),
        "test_count": selection.get("test_count"),
        "captions_per_image": selection.get("captions_per_image"),
        "seed": selection.get("seed"),
        "model": model_value,
        "batch_size": embedding.get("batch_size"),
        "device": embedding.get("device"),
        "max_size": preprocessing.get("max_size"),
        "near_duplicate_threshold": index.get("near_duplicate_threshold"),
    }.items():
        if getattr(args, name, None) is None and value is not None:
            setattr(args, name, value)


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


def image_caption_counts(captions):
    counts = Counter(
        image_id_key(annotation["image_id"]) for annotation in captions
    )
    return counts


def validate_coco_image(image, image_root):
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise RuntimeError(
            "Dataset image validation requires Pillow. Install requirements.txt."
        ) from error

    image_path = Path(image_root) / image["file_name"]
    if not image_path.is_file():
        return None, {"reason": "missing", "image_id": image_id_key(image["id"])}
    try:
        with Image.open(image_path) as source:
            source.verify()
        with Image.open(image_path) as source:
            original_width, original_height = source.size
            oriented = ImageOps.exif_transpose(source)
            dimensions = [int(oriented.width), int(oriented.height)]
    except (OSError, ValueError, SyntaxError) as error:
        return None, {
            "reason": "corrupt",
            "image_id": image_id_key(image["id"]),
            "detail": str(error),
        }
    return {
        "image_id": image_id_key(image["id"]),
        "file_name": image["file_name"],
        "image_path": str(image_path.resolve()),
        "path": str(image_path.resolve()),
        "sha256": sha256_file(image_path),
        "width": dimensions[0],
        "height": dimensions[1],
        "original_width": int(original_width),
        "original_height": int(original_height),
    }, None


def audit_image_candidates(images, image_root, candidate_count, seed):
    if candidate_count <= 0:
        raise ValueError("Candidate count must be greater than zero.")
    image_records = sorted(
        images, key=lambda row: image_id_key(row["id"])
    )
    image_ids = [image_id_key(row["id"]) for row in image_records]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Source annotations contain duplicate image IDs.")
    if len(image_records) < candidate_count:
        raise ValueError(
            f"Requested {candidate_count} candidates, but annotations contain "
            f"only {len(image_records)} images."
        )
    selected = random.Random(seed).sample(image_records, candidate_count)
    candidates = []
    excluded = []
    hashes = {}
    for image in selected:
        row, issue = validate_coco_image(image, image_root)
        if issue:
            excluded.append(issue)
            continue
        if row["sha256"] in hashes:
            excluded.append(
                {
                    "reason": "duplicate",
                    "image_id": row["image_id"],
                    "duplicate_of": hashes[row["sha256"]],
                }
            )
            continue
        hashes[row["sha256"]] = row["image_id"]
        row["width"] = int(image.get("width", row["width"]))
        row["height"] = int(image.get("height", row["height"]))
        candidates.append(row)
    return candidates, excluded


def audit_dataset(args):
    instances = load_json(args.instances)
    captions = read_coco_captions(args.captions)
    images = instances.get("images", [])
    image_ids = {image_id_key(image["id"]) for image in images}
    caption_counts = image_caption_counts(captions)
    caption_image_ids = set(caption_counts)
    annotation_image_ids = {
        image_id_key(row["image_id"])
        for row in instances.get("annotations", [])
    }
    report = {
        "version": 1,
        "source": "MS COCO train2017",
        "image_annotation_count": len(images),
        "unique_image_id_count": len(image_ids),
        "object_annotation_count": len(instances.get("annotations", [])),
        "caption_annotation_count": len(captions),
        "captioned_image_count": len(caption_image_ids),
        "images_without_captions": sorted(image_ids - caption_image_ids),
        "captions_with_unknown_image_ids": sorted(caption_image_ids - image_ids),
        "object_annotations_with_unknown_image_ids": sorted(
            annotation_image_ids - image_ids
        ),
        "caption_count_distribution": dict(
            sorted(Counter(caption_counts.values()).items())
        ),
        "passed": (
            len(images) == len(image_ids)
            and not (caption_image_ids - image_ids)
            and not (annotation_image_ids - image_ids)
        ),
    }
    if args.images:
        image_records = {image_id_key(row["id"]): row for row in images}
        available_files = {
            entry.name
            for entry in os.scandir(args.images)
            if entry.is_file(follow_symlinks=False)
        }
        missing = [
            {"image_id": image_id, "file_name": row["file_name"]}
            for image_id, row in image_records.items()
            if row["file_name"] not in available_files
        ]
        report["image_directory"] = str(Path(args.images).resolve())
        report["missing_image_count"] = len(missing)
        report["missing_images"] = missing
        report["passed"] = report["passed"] and not missing
    write_json(args.output, report)
    if not report["passed"]:
        raise ValueError(f"Dataset audit failed; see {args.output}.")
    print(f"Dataset audit passed; report written to {Path(args.output).resolve()}")


def select_candidates(args):
    instance_data = load_json(args.instances)
    images = instance_data.get("images", [])
    if not images:
        raise ValueError("No image records were found in the instances file.")
    candidates, excluded = audit_image_candidates(
        images, args.images, args.candidate_count, args.seed
    )
    captions = read_coco_captions(args.captions)
    counts = image_caption_counts(captions)
    for row in candidates:
        row["caption_count"] = counts[row["image_id"]]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_id",
        "file_name",
        "image_path",
        "path",
        "sha256",
        "width",
        "height",
        "original_width",
        "original_height",
        "processed_width",
        "processed_height",
        "caption_count",
    ]
    write_csv(output / "candidate_manifest.csv", candidates, fields)
    write_json(
        output / "candidate_audit.json",
        {
            "version": 1,
            "seed": args.seed,
            "candidate_pool_requested": args.candidate_count,
            "candidate_pool_valid_unique": len(candidates),
            "excluded_count": len(excluded),
            "excluded_by_reason": dict(
                sorted(Counter(row["reason"] for row in excluded).items())
            ),
            "excluded": excluded,
            "source_annotations": str(Path(args.instances).resolve()),
            "source_images": str(Path(args.images).resolve()),
            "caption_count_anomalies": [
                {
                    "image_id": row["image_id"],
                    "caption_count": row["caption_count"],
                }
                for row in candidates
                if row["caption_count"] != args.captions_per_image
            ],
        },
    )
    minimum = args.knowledge_base_count + args.validation_count
    if len(candidates) < minimum:
        raise ValueError(
            f"Candidate pool has {len(candidates)} valid unique images; "
            f"{minimum} are required ({args.knowledge_base_count} knowledge "
            f"base + {args.validation_count} validation). See "
            f"{output / 'candidate_audit.json'}."
        )
    print(
        f"Audited {args.candidate_count} candidates; "
        f"{len(candidates)} valid unique images are available."
    )


def source_file_checksum(path):
    return sha256_file(path)


def difference_hash(path):
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise RuntimeError(
            "Near-duplicate analysis requires Pillow. Install requirements.txt."
        ) from error
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L").resize((9, 8))
        pixels = image.load()
    value = 0
    for row in range(8):
        for column in range(8):
            left = pixels[column, row]
            right = pixels[column + 1, row]
            value = (value << 1) | int(left > right)
    return value


def near_duplicate_review_rows(splits, threshold):
    if not 0 <= threshold <= 1:
        raise ValueError("Near-duplicate threshold must be between zero and one.")
    split_names = ("knowledge_base", "validation", "test")
    split_records = {
        split_name: splits[split_name] for split_name in split_names
    }
    records = [
        row for split_name in split_names for row in split_records[split_name]
    ]
    hashes = {
        row["image_id"]: difference_hash(row["image_path"])
        for row in records
    }
    candidates = []
    for left_index, left_split in enumerate(split_names):
        for right_split in split_names[left_index + 1 :]:
            for left in split_records[left_split]:
                for right in split_records[right_split]:
                    distance = (
                        hashes[left["image_id"]] ^ hashes[right["image_id"]]
                    ).bit_count()
                    similarity = 1 - distance / 64
                    if similarity >= threshold:
                        candidates.append(
                            {
                                "image_id_a": left["image_id"],
                                "file_name_a": left["file_name"],
                                "split_a": left["split"],
                                "image_id_b": right["image_id"],
                                "file_name_b": right["file_name"],
                                "split_b": right["split"],
                                "dhash_similarity": round(similarity, 6),
                                "review_status": "pending",
                            }
                        )
    return candidates


def caption_manifest(captions, image_ids, split_name):
    selected = []
    counts = Counter()
    for annotation in captions:
        image_id = image_id_key(annotation["image_id"])
        if image_id not in image_ids:
            continue
        raw_caption = annotation["caption"]
        selected.append(
            {
                "caption_id": annotation["id"],
                "image_id": image_id,
                "raw_caption": raw_caption,
                "cleaned_caption": clean_caption(raw_caption),
                "split": split_name,
            }
        )
        counts[image_id] += 1
    missing = sorted(image_id for image_id in image_ids if counts[image_id] == 0)
    return selected, counts, missing


def freeze_splits(args):
    candidates = read_manifest(args.manifest)
    required = args.knowledge_base_count + args.validation_count
    if len(candidates) < required:
        raise ValueError(
            f"Need at least {required} valid candidate images; found "
            f"{len(candidates)}."
        )
    candidate_ids = [row["image_id"] for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate manifest contains duplicate image IDs.")
    selected = random.Random(args.seed).sample(candidates, required)
    knowledge_base = selected[: args.knowledge_base_count]
    validation = selected[args.knowledge_base_count :]
    train_ids = {row["image_id"] for row in knowledge_base}
    validation_ids = {row["image_id"] for row in validation}

    test_instances = load_json(args.test_instances)
    test_captions_source = read_coco_captions(args.test_captions)
    val_images = {
        image_id_key(image["id"]): image
        for image in test_instances.get("images", [])
    }
    if len(val_images) != len(test_instances.get("images", [])):
        raise ValueError("COCO val2017 instances contain duplicate image IDs.")
    val_captions = image_caption_counts(test_captions_source)
    unknown_test_caption_ids = sorted(set(val_captions) - set(val_images))
    if unknown_test_caption_ids:
        raise ValueError(
            "COCO val2017 captions reference image IDs absent from val "
            f"instances: {unknown_test_caption_ids[:10]}"
        )
    valid_test_images = [
        image
        for image in test_instances.get("images", [])
        if val_captions[image_id_key(image["id"])] > 0
    ]
    if len(valid_test_images) < args.test_count:
        raise ValueError(
            f"COCO val2017 annotations contain only {len(valid_test_images)} "
            f"images with captions; {args.test_count} are required."
        )
    shuffled_test_images = sorted(
        valid_test_images, key=lambda row: image_id_key(row["id"])
    )
    random.Random(args.seed).shuffle(shuffled_test_images)
    test = []
    excluded = []
    test_hashes = set()
    for image in shuffled_test_images:
        if len(test) == args.test_count:
            break
        row, issue = validate_coco_image(image, args.test_images)
        if issue:
            excluded.append(issue)
            continue
        if row["image_id"] in train_ids | validation_ids:
            raise ValueError(
                f"COCO val2017 image ID overlaps train2017: {row['image_id']}."
            )
        if row["sha256"] in test_hashes:
            excluded.append(
                {"reason": "duplicate", "image_id": row["image_id"]}
            )
            continue
        test_hashes.add(row["sha256"])
        row["width"] = int(image.get("width", row["width"]))
        row["height"] = int(image.get("height", row["height"]))
        test.append(row)
    if len(test) != args.test_count:
        raise ValueError(
            f"Only {len(test)} readable, unique val2017 images were selected; "
            f"{args.test_count} are required. Exclusions: {excluded[:10]}"
        )

    train_hashes = {row["sha256"] for row in knowledge_base}
    validation_hashes = {row["sha256"] for row in validation}
    exact_overlap = {
        "knowledge_base_validation": sorted(train_hashes & validation_hashes),
        "knowledge_base_test": sorted(train_hashes & test_hashes),
        "validation_test": sorted(validation_hashes & test_hashes),
    }
    if any(exact_overlap.values()):
        raise ValueError(f"Exact duplicate image content across splits: {exact_overlap}")

    train_captions_source = read_coco_captions(args.captions)
    kb_captions, kb_caption_counts, kb_missing = caption_manifest(
        train_captions_source, train_ids, "knowledge_base"
    )
    validation_captions, validation_caption_counts, validation_missing = (
        caption_manifest(train_captions_source, validation_ids, "validation")
    )
    test_ids = {row["image_id"] for row in test}
    test_captions, test_caption_counts, test_missing = caption_manifest(
        test_captions_source, test_ids, "test"
    )
    test_reference_count = len(test_captions)
    test_caption_count_mismatches = [
        {
            "image_id": image_id,
            "caption_count": test_caption_counts[image_id],
            "expected": args.captions_per_image,
        }
        for image_id in sorted(test_ids)
        if test_caption_counts[image_id] != args.captions_per_image
    ]
    kb_caption_count_mismatches = [
        {
            "image_id": image_id,
            "caption_count": kb_caption_counts[image_id],
            "expected": args.captions_per_image,
        }
        for image_id in sorted(train_ids)
        if kb_caption_counts[image_id] != args.captions_per_image
    ]
    validation_caption_count_mismatches = [
        {
            "image_id": image_id,
            "caption_count": validation_caption_counts[image_id],
            "expected": args.captions_per_image,
        }
        for image_id in sorted(validation_ids)
        if validation_caption_counts[image_id] != args.captions_per_image
    ]
    caption_blockers = {
        "knowledge_base_missing_captions": kb_missing,
        "validation_missing_captions": validation_missing,
        "test_missing_captions": test_missing,
        "test_caption_count_mismatches": test_caption_count_mismatches,
        "expected_test_reference_count": args.test_count
        * args.captions_per_image,
        "actual_test_reference_count": test_reference_count,
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "splits_v2.json"
    version = 2
    if manifest_path.exists():
        if not args.force_new_splits:
            raise FileExistsError(
                f"Frozen split manifest already exists: {manifest_path}. "
                "Use --force-new-splits to create and preserve a new version."
            )
        previous = load_json(manifest_path)
        version = int(previous.get("manifest_version", 2)) + 1
        history = output / "history" / f"v{previous.get('manifest_version', 2)}"
        history.mkdir(parents=True, exist_ok=True)
        previous_files = ["splits_v2.json"] + list(
            previous.get("manifest_files", {}).keys()
        )
        for name in previous_files:
            source = output / name
            archived = history / Path(name).name
            if source.is_file():
                if archived.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite previous split archive: {archived}"
                    )
                shutil.copy2(source, archived)

    knowledge_base = [
        {**row, "split": "knowledge_base"} for row in knowledge_base
    ]
    validation = [{**row, "split": "validation"} for row in validation]
    test = [{**row, "split": "test"} for row in test]
    reserve = [
        {**row, "split": "reserve"}
        for row in candidates
        if row["image_id"] not in train_ids | validation_ids
    ]
    near_duplicates = near_duplicate_review_rows(
        {
            "knowledge_base": knowledge_base,
            "validation": validation,
            "test": test,
        },
        args.near_duplicate_threshold,
    )
    manifest_fields = [
        "image_id",
        "file_name",
        "image_path",
        "path",
        "split",
        "sha256",
        "width",
        "height",
        "original_width",
        "original_height",
        "processed_width",
        "processed_height",
        "caption_count",
    ]
    for name, rows in (
        ("knowledge_base_manifest.csv", knowledge_base),
        ("validation_manifest.csv", validation),
        ("test_manifest.csv", test),
        ("reserve_manifest.csv", reserve),
    ):
        write_csv(output / name, rows, manifest_fields)
    write_csv(output / "knowledge_base.csv", knowledge_base, manifest_fields)
    write_csv(output / "train.csv", knowledge_base, manifest_fields)
    write_csv(output / "validation.csv", validation, manifest_fields)
    write_csv(output / "test.csv", test, manifest_fields)
    for name, rows in (
        ("knowledge_base_captions.csv", kb_captions),
        ("validation_captions.csv", validation_captions),
        ("test_captions.csv", test_captions),
    ):
        write_csv(
            output / name,
            rows,
            ["caption_id", "image_id", "raw_caption", "cleaned_caption", "split"],
        )
    write_csv(
        output / "train_captions.csv",
        kb_captions,
        ["caption_id", "image_id", "raw_caption", "cleaned_caption", "split"],
    )
    write_csv(
        output / "near_duplicate_review.csv",
        near_duplicates,
        [
            "image_id_a",
            "file_name_a",
            "split_a",
            "image_id_b",
            "file_name_b",
            "split_b",
            "dhash_similarity",
            "review_status",
        ],
    )
    annotation_paths = {
        "train_instances": Path(args.instances).resolve(),
        "train_captions": Path(args.captions).resolve(),
        "val_instances": Path(args.test_instances).resolve(),
        "val_captions": Path(args.test_captions).resolve(),
    }
    leakage_audit = {
        "knowledge_base_validation_ids_disjoint": not train_ids
        & validation_ids,
        "knowledge_base_test_ids_disjoint": not train_ids & test_ids,
        "validation_test_ids_disjoint": not validation_ids & test_ids,
        "knowledge_base_validation_hashes_disjoint": not exact_overlap[
            "knowledge_base_validation"
        ],
        "knowledge_base_test_hashes_disjoint": not exact_overlap[
            "knowledge_base_test"
        ],
        "validation_test_hashes_disjoint": not exact_overlap["validation_test"],
        "exact_duplicate_image_ids_by_split": exact_overlap,
        "near_duplicate_review_pending_count": len(near_duplicates),
        "near_duplicate_similarity_threshold": args.near_duplicate_threshold,
    }
    payload = {
        "manifest_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": args.seed,
        "source_dataset": "MS COCO 2017",
        "source_annotations": {
            key: str(path) for key, path in annotation_paths.items()
        },
        "source_annotation_sha256": {
            key: source_file_checksum(path)
            for key, path in annotation_paths.items()
        },
        "selection_policy": {
            "knowledge_base": "Random unique train2017 candidates",
            "validation": "Disjoint train2017 candidates",
            "test": "Random val2017 images with caption references",
            "test_reference_policy": "Five original references expected per image; raw text preserved.",
        },
        "counts": {
            "candidate_pool_valid_unique": len(candidates),
            "knowledge_base_images": len(knowledge_base),
            "validation_images": len(validation),
            "test_images": len(test),
            "reserve_images": len(reserve),
            "near_duplicate_pairs_flagged": len(near_duplicates),
            "knowledge_base_captions": len(kb_captions),
            "validation_captions": len(validation_captions),
            "test_captions": test_reference_count,
        },
        "caption_integrity": caption_blockers,
        "caption_count_anomalies": {
            "knowledge_base": kb_caption_count_mismatches,
            "validation": validation_caption_count_mismatches,
            "test": test_caption_count_mismatches,
        },
        "leakage_audit": leakage_audit,
        "manifest_files": {
            name: str(output / name)
            for name in (
                "knowledge_base_manifest.csv",
                "validation_manifest.csv",
                "test_manifest.csv",
                "reserve_manifest.csv",
                "knowledge_base.csv",
                "train.csv",
                "validation.csv",
                "test.csv",
                "knowledge_base_captions.csv",
                "validation_captions.csv",
                "test_captions.csv",
                "train_captions.csv",
                "near_duplicate_review.csv",
            )
        },
    }
    payload["manifest_file_sha256"] = {
        name: sha256_file(output / name)
        for name in payload["manifest_files"]
    }
    write_json(manifest_path, payload)
    payload["split_manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    write_json(manifest_path, payload)
    if any(caption_blockers[key] for key in (
        "knowledge_base_missing_captions",
        "validation_missing_captions",
        "test_missing_captions",
        "test_caption_count_mismatches",
    )) or test_reference_count != args.test_count * args.captions_per_image:
        raise ValueError(
            f"Caption integrity is blocking acceptance; see {manifest_path}."
        )
    print(
        f"Frozen splits v{version}: KB={len(knowledge_base)}, "
        f"validation={len(validation)}, test={len(test)}, "
        f"test references={test_reference_count}."
    )


def image_statistics(images, instances, captions):
    objects_by_image = Counter(
        image_id_key(annotation["image_id"]) for annotation in instances
    )
    captions_by_image = defaultdict(list)
    for annotation in captions:
        captions_by_image[image_id_key(annotation["image_id"])].append(
            annotation["caption"]
        )

    rows = []
    for image in sorted(images, key=lambda item: str(item["id"])):
        image_id = image_id_key(image["id"])
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
    if getattr(args, "manifest", None):
        selected_ids = {
            row["image_id"] for row in read_manifest(args.manifest)
        }
        images = [
            image for image in images
            if image_id_key(image["id"]) in selected_ids
        ]
        instances = [
            row for row in instances
            if image_id_key(row["image_id"]) in selected_ids
        ]
        captions = [
            row for row in captions
            if image_id_key(row["image_id"]) in selected_ids
        ]
        if len(images) != len(selected_ids):
            raise ValueError(
                "EDA manifest IDs do not all exist in the source annotations."
            )
    categories = {
        category["id"]: category["name"]
        for category in instance_data.get("categories", [])
    }

    category_counts = Counter(
        categories.get(annotation["category_id"], str(annotation["category_id"]))
        for annotation in instances
    )
    category_image_counts = Counter()
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
    for image_categories in categories_by_image.values():
        category_image_counts.update(image_categories)
    for image in images:
        for key in ("scene", "context", "location", "weather", "timeofday"):
            value = image.get(key)
            if value:
                context_counts[f"{key}:{value}"] += 1

    rows = image_statistics(images, instances, captions)
    caption_lengths = [
        len(re.findall(r"\b[\w'-]+\b", annotation["caption"]))
        for annotation in captions
    ]
    tokens = [
        token.lower()
        for annotation in captions
        for token in re.findall(r"\b[\w'-]+\b", annotation["caption"])
    ]
    object_counts = [row["object_count"] for row in rows]
    caption_counts = [row["caption_count"] for row in rows]
    sorted_object_counts = sorted(object_counts)
    caption_length_stats = {
        "mean": statistics.mean(caption_lengths) if caption_lengths else 0,
        "median": statistics.median(caption_lengths) if caption_lengths else 0,
        "stdev": statistics.stdev(caption_lengths) if len(caption_lengths) > 1 else 0,
        "min": min(caption_lengths, default=0),
        "max": max(caption_lengths, default=0),
        "percentiles": {
            str(percentile): (
                sorted(caption_lengths)[
                    min(
                        len(caption_lengths) - 1,
                        round((len(caption_lengths) - 1) * percentile / 100),
                    )
                ]
                if caption_lengths
                else 0
            )
            for percentile in (25, 50, 75, 90, 95)
        },
    }
    object_count_stats = {
        "mean": statistics.mean(object_counts) if object_counts else 0,
        "median": statistics.median(object_counts) if object_counts else 0,
        "stdev": statistics.stdev(object_counts) if len(object_counts) > 1 else 0,
        "min": min(object_counts, default=0),
        "max": max(object_counts, default=0),
        "zero_object_image_count": sum(count == 0 for count in object_counts),
        "percentiles": {
            str(percentile): (
                sorted_object_counts[
                    min(
                        len(sorted_object_counts) - 1,
                        round(
                            (len(sorted_object_counts) - 1)
                            * percentile
                            / 100
                        ),
                    )
                ]
                if sorted_object_counts
                else 0
            )
            for percentile in (25, 50, 75, 90, 95)
        },
    }
    scene_complexity = Counter(
        "0" if count == 0 else "1-3" if count <= 3 else "4-6" if count <= 6 else "7+"
        for count in object_counts
    )
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
        "caption_length_statistics": caption_length_stats,
        "objects_per_image_statistics": object_count_stats,
        "scene_complexity_is_object_count_proxy": True,
        "vocabulary_size": len(set(tokens)),
        "pos_analysis": pos_status,
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    write_csv(
        output / "categories.csv",
        [
            {
                "category": category,
                "object_count": count,
                "image_count": category_image_counts[category],
            }
            for category, count in sorted(category_counts.items())
        ],
        ["category", "object_count", "image_count"],
    )
    write_csv(
        output / "top_categories.csv",
        [
            {
                "category": category,
                "object_count": count,
                "image_count": category_image_counts[category],
            }
            for category, count in category_counts.most_common(50)
        ],
        ["category", "object_count", "image_count"],
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
    write_csv(
        output / "scene_complexity.csv",
        [
            {"object_count_range": label, "image_count": scene_complexity[label]}
            for label in ("0", "1-3", "4-6", "7+")
        ],
        ["object_count_range", "image_count"],
    )
    write_csv(
        output / "caption_color_mentions.csv",
        caption_lexicon_counts(captions, (
            "black", "white", "gray", "grey", "red", "orange", "yellow",
            "green", "blue", "purple", "pink", "brown",
        )),
        ["term", "caption_count"],
    )
    write_csv(
        output / "caption_context_mentions.csv",
        caption_lexicon_counts(captions, (
            "indoor", "indoors", "outdoor", "outdoors", "day", "daytime",
            "night", "nighttime", "sunny", "rainy", "snowy", "beach",
            "street", "kitchen", "park", "room",
        )),
        ["term", "caption_count"],
    )
    write_json(
        output / "text_statistics.json",
        {
            "caption_count": len(captions),
            "vocabulary_size": len(set(tokens)),
            "mean_caption_words": summary["mean_caption_words"],
            "pos_analysis": pos_status,
            "object_count_statistics": object_count_stats,
            "caption_length_statistics": caption_length_stats,
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
    create_eda_visualizations(
        output,
        category_counts,
        category_image_counts,
        cooccurrence,
        tokens,
        pos_counts,
        rows,
        scene_complexity,
        captions,
    )
    write_json(
        output / "analysis_availability.json",
        {
            "cooccurrence": "Derived from COCO object categories per image.",
            "spatial_relationships": (
                "COCO instances do not supply semantic relationships; "
                "only source-provided relationship fields are summarized."
            ),
            "attributes_and_context": (
                "Only included when present in source annotations; "
                "caption mentions are descriptive text, not image labels."
            ),
            "color_shape": (
                "Color terms are counted from captions; no color or shape "
                "is inferred to be visually present."
            ),
            "pos": pos_status,
        },
    )
    print(f"Exploration reports written to {output.resolve()}")


def caption_lexicon_counts(captions, terms):
    normalized_terms = {term.lower() for term in terms}
    counts = Counter()
    for annotation in captions:
        tokens_in_caption = set(
            token.lower()
            for token in re.findall(r"\b[\w'-]+\b", annotation["caption"])
        )
        counts.update(tokens_in_caption.intersection(normalized_terms))
    return [
        {"term": term, "caption_count": counts[term]}
        for term in sorted(normalized_terms)
    ]


def create_eda_visualizations(
    output,
    category_counts,
    category_image_counts,
    cooccurrence,
    tokens,
    pos_counts,
    rows,
    scene_complexity,
    captions,
):
    import matplotlib.pyplot as plt

    top_categories = category_counts.most_common(50)
    figure, axis = plt.subplots(figsize=(12, 10))
    axis.barh(
        [category for category, _ in reversed(top_categories)],
        [count for _, count in reversed(top_categories)],
    )
    axis.set(title="Top 50 COCO object categories", xlabel="Annotations")
    figure.tight_layout()
    figure.savefig(output / "top_50_object_categories.png", dpi=200)
    plt.close(figure)

    frequent_categories = [
        category for category, _ in category_counts.most_common(25)
    ]
    figure, axis = plt.subplots(figsize=(12, 10))
    matrix = [
        [
            cooccurrence.get(tuple(sorted((left, right))), 0)
            if left != right
            else category_image_counts[left]
            for right in frequent_categories
        ]
        for left in frequent_categories
    ]
    image = axis.imshow(matrix, cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(frequent_categories)), frequent_categories, rotation=90)
    axis.set_yticks(range(len(frequent_categories)), frequent_categories)
    axis.set_title("Object-category co-occurrence (images)")
    figure.colorbar(image, ax=axis, label="Images")
    figure.tight_layout()
    figure.savefig(output / "object_cooccurrence_heatmap.png", dpi=200)
    plt.close(figure)

    vocabulary = Counter(tokens).most_common(30)
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(
        [token for token, _ in vocabulary],
        [count for _, count in vocabulary],
    )
    axis.set(title="Most frequent caption words", ylabel="Token count")
    axis.tick_params(axis="x", labelrotation=60)
    figure.tight_layout()
    figure.savefig(output / "caption_vocabulary_frequency.png", dpi=200)
    plt.close(figure)

    if pos_counts:
        figure, axis = plt.subplots(figsize=(9, 6))
        ordered_pos = sorted(pos_counts.items(), key=lambda row: row[1])
        axis.barh(
            [label for label, _ in ordered_pos],
            [count for _, count in ordered_pos],
        )
        axis.set(title="Part-of-speech distribution", xlabel="Token count")
        figure.tight_layout()
        figure.savefig(output / "pos_distribution.png", dpi=200)
        plt.close(figure)

    complexity_order = ("0", "1-3", "4-6", "7+")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        complexity_order,
        [scene_complexity[label] for label in complexity_order],
    )
    axis.set(
        title="Scene complexity proxy (object annotations per image)",
        xlabel="Objects per image",
        ylabel="Images",
    )
    figure.tight_layout()
    figure.savefig(output / "scene_complexity.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(
        [row["object_count"] for row in rows],
        [row["average_caption_words"] for row in rows],
        alpha=0.35,
        s=12,
    )
    axis.set(
        title="Object count versus mean caption length",
        xlabel="Object annotations per image",
        ylabel="Mean words per caption",
    )
    figure.tight_layout()
    figure.savefig(output / "objects_vs_caption_length.png", dpi=200)
    plt.close(figure)

    for filename, title, terms in (
        (
            "caption_color_mentions.png",
            "Caption-mentioned color terms (not visual labels)",
            (
                "black", "white", "gray", "grey", "red", "orange", "yellow",
                "green", "blue", "purple", "pink", "brown",
            ),
        ),
        (
            "caption_context_mentions.png",
            "Caption-mentioned context terms (not visual labels)",
            (
                "indoor", "indoors", "outdoor", "outdoors", "day", "daytime",
                "night", "nighttime", "sunny", "rainy", "snowy", "beach",
                "street", "kitchen", "park", "room",
            ),
        ),
    ):
        mention_rows = caption_lexicon_counts(captions, terms)
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.bar(
            [row["term"] for row in mention_rows],
            [row["caption_count"] for row in mention_rows],
        )
        axis.set(title=title, ylabel="Captions mentioning term")
        axis.tick_params(axis="x", labelrotation=45)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=200)
        plt.close(figure)


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
                "category_id": annotation.get("category_id"),
                "annotation_id": annotation.get("id"),
                "bbox": annotation.get("bbox"),
                "area": annotation.get("area"),
                "segmentation": annotation.get("segmentation"),
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
                "width",
                "height",
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
                "object_count": len(objects),
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
    kb_manifest = split_root / "knowledge_base_manifest.csv"
    if not kb_manifest.is_file():
        kb_manifest = split_root / "knowledge_base.csv"
    manifests = {
        "knowledge_base": read_manifest(kb_manifest),
        "validation": read_manifest(split_root / "validation.csv")
        if (split_root / "validation.csv").is_file()
        else read_manifest(split_root / "validation_manifest.csv"),
        "test": read_manifest(split_root / "test.csv")
        if (split_root / "test.csv").is_file()
        else read_manifest(split_root / "test_manifest.csv"),
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
    knowledge_base_ids = {row["image_id"] for row in manifests["knowledge_base"]}
    if not knowledge_base_ids.issubset(known_ids):
        raise ValueError(
            "The instances annotations do not contain every knowledge-base image."
        )
    train_ids = knowledge_base_ids
    validation_ids = {row["image_id"] for row in manifests["validation"]}
    test_ids = {row["image_id"] for row in manifests["test"]}
    if (train_ids & validation_ids) or (train_ids & test_ids) or (
        validation_ids & test_ids
    ):
        raise ValueError("Knowledge-base, validation, and test IDs must be disjoint.")
    knowledge_base_hashes = {
        row["sha256"] for row in manifests["knowledge_base"]
    }
    validation_hashes = {
        row["sha256"] for row in manifests["validation"]
    }
    test_hashes = {row["sha256"] for row in manifests["test"]}
    if (
        knowledge_base_hashes.intersection(validation_hashes)
        or knowledge_base_hashes.intersection(test_hashes)
        or validation_hashes.intersection(test_hashes)
    ):
        raise ValueError("Exact duplicate image content exists across splits.")

    output = Path(args.output).resolve()
    image_output = output / "images"
    image_output.mkdir(parents=True, exist_ok=True)
    processed_by_id = {}
    seen_targets = set()
    for name in ("knowledge_base", "validation", "test"):
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
            if target.exists():
                raise FileExistsError(
                    f"Refusing to overwrite processed image: {target}"
                )
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                if args.max_size > 0 and max(image.size) > args.max_size:
                    image.thumbnail(
                        (args.max_size, args.max_size),
                        Image.Resampling.LANCZOS,
                    )
                image.save(target, format="JPEG", quality=95)
                processed_width, processed_height = image.size
                image.close()
            processed = dict(row)
            processed["path"] = str(target.resolve())
            processed["sha256"] = sha256_file(target)
            processed["processed_width"] = processed_width
            processed["processed_height"] = processed_height
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
        fields = list(records[0]) + ["processed_width", "processed_height"]
        write_csv(output / f"{name}.csv", converted, fields)
    write_csv(
        output / "train.csv",
        [
            processed_by_id[row["image_id"]]
            for row in manifests["knowledge_base"]
        ],
        list(manifests["knowledge_base"][0])
        + ["processed_width", "processed_height"],
    )

    captions = read_coco_captions(args.captions)
    train_caption_rows, train_mismatches, train_distribution = caption_rows(
        captions, train_ids, args.captions_per_image
    )
    write_csv(
        output / "train_captions.csv",
        train_caption_rows,
        ["caption_id", "image_id", "raw_caption", "cleaned_caption"],
    )
    write_csv(
        output / "knowledge_base_captions.csv",
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
                **image,
                "image_id": image["id"],
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
                "object_count": row["object_count"],
                "objects": json.dumps(row["objects"], ensure_ascii=False),
                "attributes": json.dumps(row["attributes"], ensure_ascii=False),
                "relationships": json.dumps(
                    row["relationships"], ensure_ascii=False
                ),
                "context": json.dumps(row["context"], ensure_ascii=False),
            }
            for row in metadata
        ],
        [
            "image_id",
            "object_count",
            "objects",
            "attributes",
            "relationships",
            "context",
        ],
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


def embed_sbert(args):
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Sentence-BERT embeddings require sentence-transformers and numpy. "
            "Install requirements-embeddings.txt."
        ) from error

    records = read_manifest(args.manifest)
    if not args.captions:
        raise ValueError("Sentence-BERT embedding requires --captions.")
    caption_records = load_caption_records(args.captions)
    record_ids = {row["image_id"] for row in records}
    outside_ids = {row["image_id"] for row in caption_records} - record_ids
    if outside_ids:
        raise ValueError(
            "Caption input includes image IDs outside the embedding manifest "
            f"(possible leakage): {sorted(outside_ids)[:5]}"
        )
    if not caption_records:
        raise ValueError("Caption manifest contains no captions to embed.")

    model_name = args.model or "sentence-transformers/all-MiniLM-L6-v2"
    device = args.device if args.device not in (None, "auto") else None
    model = SentenceTransformer(model_name, device=device)
    started = time.perf_counter()
    embeddings = model.encode(
        [row["cleaned_caption"] for row in caption_records],
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    dimension = int(embeddings.shape[1])
    if dimension != 384:
        raise ValueError(
            f"Expected 384-dimensional Sentence-BERT vectors, got {dimension}."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Sentence-BERT produced non-finite vectors.")

    output = Path(args.output)
    if output.suffix.lower() != ".npz":
        raise ValueError("Embedding output path must use the .npz extension.")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        image_ids=np.asarray([row["image_id"] for row in records]),
        image_embeddings=np.empty((0, dimension), dtype=np.float32),
        embeddings=np.empty((0, dimension), dtype=np.float32),
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
        text_embeddings=embeddings,
    )
    write_json(
        output.with_suffix(".json"),
        {
            "model": model_name,
            "count": len(embeddings),
            "dimension": dimension,
            "normalized": True,
            "similarity_metric": "cosine",
            "batch_size": args.batch_size,
            "device": str(model.device),
            "sentence_transformers_version": __import__(
                "sentence_transformers"
            ).__version__,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "manifest": str(Path(args.manifest).resolve()),
            "captions": str(Path(args.captions).resolve()),
        },
    )
    print(f"Saved {len(embeddings)} Sentence-BERT caption vectors to {output}")


def load_caption_records(path):
    caption_path = Path(path)
    if caption_path.suffix.lower() == ".csv":
        with caption_path.open("r", encoding="utf-8", newline="") as file:
            caption_records = list(csv.DictReader(file))
        for index, row in enumerate(caption_records):
            if not str(row.get("image_id") or "").strip():
                raise ValueError("Caption CSV contains a row without an image_id.")
            if not str(row.get("caption_id") or "").strip():
                raise ValueError("Caption CSV contains a row without a caption_id.")
            row["caption_id"] = str(row["caption_id"])
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
        return caption_records

    return [
        {
            "caption_id": str(annotation.get("id", index)),
            "image_id": image_id_key(annotation["image_id"]),
            "raw_caption": annotation["caption"],
            "cleaned_caption": clean_caption(annotation["caption"]),
        }
        for index, annotation in enumerate(read_coco_captions(caption_path))
    ]


def embed(args):
    if args.encoder == "sbert":
        return embed_sbert(args)
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
        caption_records = load_caption_records(args.captions)
        outside_ids = {
            row["image_id"] for row in caption_records
        } - record_ids
        if outside_ids:
            raise ValueError(
                "Caption input includes image IDs outside the embedding "
                f"manifest (possible split leakage): {sorted(outside_ids)[:5]}"
            )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model or "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
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
            "model": model_name,
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


def normalized_matrix(matrix, label, expected_dimension):
    import numpy as np

    if matrix.ndim != 2 or matrix.shape[1] != expected_dimension:
        raise ValueError(
            f"{label} must have shape (n, {expected_dimension}); got {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} contains NaN or infinite values.")
    if len(matrix) == 0:
        raise ValueError(f"{label} is empty.")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1, atol=1e-3):
        raise ValueError(f"{label} is not L2-normalized.")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def build_faiss_indexes(args):
    try:
        import faiss
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "FAISS indexing requires faiss-cpu and numpy. "
            "Install requirements-faiss.txt."
        ) from error

    clip = load_embeddings(args.clip_embeddings)
    sbert = load_embeddings(args.sbert_embeddings)
    image_vectors = normalized_matrix(
        clip["image_embeddings"], "CLIP image vectors", 512
    )
    clip_caption_vectors = normalized_matrix(
        clip["text_embeddings"], "CLIP caption vectors", 512
    )
    sbert_caption_vectors = normalized_matrix(
        sbert["text_embeddings"], "Sentence-BERT caption vectors", 384
    )
    clip_caption_keys = list(
        zip(clip["caption_image_ids"].tolist(), clip["caption_ids"].tolist())
    )
    sbert_caption_keys = list(
        zip(sbert["caption_image_ids"].tolist(), sbert["caption_ids"].tolist())
    )
    if clip_caption_keys != sbert_caption_keys:
        raise ValueError(
            "CLIP and Sentence-BERT caption IDs/order do not match."
        )
    if clip["captions"].tolist() != sbert["captions"].tolist():
        raise ValueError("CLIP and Sentence-BERT raw captions do not match.")
    if not set(clip["caption_image_ids"]).issubset(set(clip["image_ids"])):
        raise ValueError("A caption vector is not linked to a knowledge-base image.")
    if not set(sbert["caption_image_ids"]).issubset(set(clip["image_ids"])):
        raise ValueError("A Sentence-BERT caption is outside the knowledge base.")
    if len(set(clip["image_ids"])) != len(clip["image_ids"]):
        raise ValueError("CLIP image vectors contain duplicate image IDs.")
    if len(set(clip_caption_keys)) != len(clip_caption_keys):
        raise ValueError("Caption embeddings contain duplicate image/caption IDs.")

    metadata_rows = load_json(args.metadata)
    metadata_by_id = {
        image_id_key(row["image_id"]): row for row in metadata_rows
    }
    if len(metadata_by_id) != len(metadata_rows):
        raise ValueError("Image metadata contains duplicate image IDs.")
    missing_metadata = sorted(
        set(clip["image_ids"].tolist()) - set(metadata_by_id)
    )
    if missing_metadata:
        raise ValueError(
            f"Image metadata is missing IDs: {missing_metadata[:10]}"
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model_metadata = {}
    for embedding_path, key in (
        (args.clip_embeddings, "clip"),
        (args.sbert_embeddings, "sbert"),
    ):
        model_path = Path(embedding_path).with_suffix(".json")
        model_metadata[key] = load_json(model_path) if model_path.is_file() else {}
    datasets = (
        (
            "clip_images",
            image_vectors,
            [
                {
                    "vector_id": f"image:{image_id}",
                    "image_id": image_id,
                    "caption_id": None,
                    "modality": "image",
                    "caption": None,
                    "metadata": metadata_by_id[image_id],
                }
                for image_id in clip["image_ids"]
            ],
            512,
        ),
        (
            "clip_captions",
            clip_caption_vectors,
            [
                {
                    "vector_id": f"caption:{image_id}:{caption_id}",
                    "image_id": image_id,
                    "caption_id": caption_id,
                    "modality": "text",
                    "caption": caption,
                    "metadata": metadata_by_id[image_id],
                }
                for image_id, caption_id, caption in zip(
                    clip["caption_image_ids"],
                    clip["caption_ids"],
                    clip["captions"],
                )
            ],
            512,
        ),
        (
            "sbert_captions",
            sbert_caption_vectors,
            [
                {
                    "vector_id": f"caption:{image_id}:{caption_id}",
                    "image_id": image_id,
                    "caption_id": caption_id,
                    "modality": "text",
                    "caption": caption,
                    "metadata": metadata_by_id[image_id],
                }
                for image_id, caption_id, caption in zip(
                    sbert["caption_image_ids"],
                    sbert["caption_ids"],
                    sbert["captions"],
                )
            ],
            384,
        ),
    )
    index_reports = {}
    for name, vectors, mapping, dimension in datasets:
        if len(mapping) != len(vectors):
            raise ValueError(f"{name} mapping count differs from vector count.")
        vector_ids = [row["vector_id"] for row in mapping]
        if len(vector_ids) != len(set(vector_ids)):
            raise ValueError(f"{name} mapping contains duplicate vector IDs.")
        index_path = output / f"{name}.faiss"
        mapping_path = output / f"{name}.mapping.json"
        if not args.overwrite and (index_path.exists() or mapping_path.exists()):
            raise FileExistsError(
                f"Refusing to overwrite an existing FAISS artifact for {name}."
            )
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        faiss.write_index(index, str(index_path))
        reopened = faiss.read_index(str(index_path))
        if reopened.ntotal != len(mapping):
            raise ValueError(f"{name} index count changed when reloaded.")
        write_json(
            mapping_path,
            {
                "version": 1,
                "index_name": name,
                "dimension": dimension,
                "similarity_metric": "cosine (inner product on L2-normalized vectors)",
                "model": model_metadata[
                    "clip" if name.startswith("clip_") else "sbert"
                ],
                "source_embeddings": str(
                    Path(
                        args.clip_embeddings
                        if name.startswith("clip_")
                        else args.sbert_embeddings
                    ).resolve()
                ),
                "vectors": mapping,
            },
        )
        index_reports[name] = {
            "index_path": str(index_path.resolve()),
            "mapping_path": str(mapping_path.resolve()),
            "vector_count": int(reopened.ntotal),
            "dimension": int(reopened.d),
        }
    write_json(
        output / "index_manifest.json",
        {
            "version": 1,
            "index_type": "faiss.IndexFlatIP",
            "models": model_metadata,
            "indexes": index_reports,
        },
    )
    print(f"Built and reopened three FAISS indexes in {output.resolve()}")


def retrieve_top_k(query, vectors, records, top_k, excluded_image_ids=()):
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Retrieval requires numpy. Install requirements-embeddings.txt."
        ) from error
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != len(query):
        raise ValueError("Query and index vector dimensions do not match.")
    if len(records) != len(vectors):
        raise ValueError("Vector metadata count does not match the index.")
    if not len(vectors):
        raise ValueError("Cannot retrieve from an empty index.")
    if not np.isfinite(query).all() or not np.isfinite(vectors).all():
        raise ValueError("Query or index contains non-finite values.")
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0:
        raise ValueError("Query vector cannot have zero length.")
    query = query / query_norm
    scores = vectors @ query
    excluded = {image_id_key(value) for value in excluded_image_ids}
    order = sorted(
        range(len(scores)),
        key=lambda index: (
            -float(scores[index]),
            str(records[index].get("vector_id", index)),
        ),
    )
    results = []
    for index in order:
        record = records[index]
        if image_id_key(record["image_id"]) in excluded:
            continue
        results.append({**record, "score": float(scores[index])})
        if len(results) == top_k:
            break
    return results


def encode_query(args):
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Query encoding requires numpy. Install requirements-embeddings.txt."
        ) from error

    if args.encoder == "sbert-text":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Sentence-BERT queries require sentence-transformers."
            ) from error
        model_name = args.model or "sentence-transformers/all-MiniLM-L6-v2"
        device = args.device if args.device not in (None, "auto") else None
        model = SentenceTransformer(model_name, device=device)
        vector = model.encode(
            [clean_caption(args.text)],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        dimension = 384
    else:
        try:
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:
            raise RuntimeError(
                "CLIP queries require torch, Pillow, and transformers."
            ) from error
        model_name = args.model or "openai/clip-vit-base-patch32"
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device)
        model.eval()
        if args.encoder == "clip-image":
            with Image.open(args.image) as source:
                image = source.convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            image.close()
            features_function = model.get_image_features
        else:
            inputs = processor(
                text=[clean_caption(args.text)],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            features_function = model.get_text_features
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            features = features_function(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            vector = torch.nn.functional.normalize(
                features, p=2, dim=-1
            )[0].cpu().numpy()
        dimension = 512
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError(
            f"Query encoder returned an invalid vector; expected ({dimension},)."
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, vector)
    write_json(
        output.with_suffix(".json"),
        {
            "encoder": args.encoder,
            "model": model_name,
            "dimension": dimension,
            "normalized": True,
            "device": (
                str(model.device)
                if args.encoder == "sbert-text"
                else device
            ),
        },
    )
    print(f"Saved {args.encoder} query vector to {output.resolve()}")


def retrieve_faiss(args):
    try:
        import faiss
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "FAISS retrieval requires faiss-cpu and numpy. "
            "Install requirements-faiss.txt."
        ) from error
    index = faiss.read_index(str(args.index))
    mapping = load_json(args.mapping)
    records = mapping.get("vectors", [])
    if index.ntotal != len(records):
        raise ValueError("FAISS index and vector mapping counts differ.")
    if len({row.get("vector_id") for row in records}) != len(records):
        raise ValueError("FAISS vector mapping contains duplicate IDs.")
    query = np.load(args.query_vector, allow_pickle=False).astype(np.float32)
    if query.shape != (index.d,):
        raise ValueError(
            f"Query dimension {query.shape} does not match index dimension {index.d}."
        )
    if not np.isfinite(query).all() or np.linalg.norm(query) == 0:
        raise ValueError("Query vector must be finite and nonzero.")
    query /= np.linalg.norm(query)
    scores, ids = index.search(query.reshape(1, -1), index.ntotal)
    excluded = {image_id_key(value) for value in args.exclude_image_id}
    results = []
    for score, index_id in zip(scores[0], ids[0]):
        if index_id < 0:
            continue
        row = records[int(index_id)]
        if image_id_key(row["image_id"]) in excluded:
            continue
        if not math.isfinite(float(score)):
            raise ValueError("FAISS returned a non-finite similarity score.")
        results.append({**row, "score": float(score)})
        if len(results) == args.top_k:
            break
    output = Path(args.output)
    write_json(
        output,
        {
            "index": str(Path(args.index).resolve()),
            "query_vector": str(Path(args.query_vector).resolve()),
            "top_k": args.top_k,
            "excluded_image_ids": sorted(excluded),
            "results": results,
        },
    )
    print(f"Retrieved {len(results)} results to {output.resolve()}")


def dataset_quality_check(args):
    split_directory = Path(args.splits)
    split_path = (
        split_directory
        if split_directory.is_file()
        else split_directory / "splits_v2.json"
    )
    split_manifest = load_json(split_path)
    expected_counts = {
        "knowledge_base": 10000,
        "validation": 100,
        "test": 100,
    }
    manifest_names = {
        "knowledge_base": "knowledge_base_manifest.csv",
        "validation": "validation_manifest.csv",
        "test": "test_manifest.csv",
    }
    processed_names = {
        "knowledge_base": "train.csv",
        "validation": "validation.csv",
        "test": "test.csv",
    }
    issues = []
    split_ids = {}
    processed_rows = {}
    for split_name, expected_count in expected_counts.items():
        source_manifest_path = Path(
            split_manifest["manifest_files"][manifest_names[split_name]]
        )
        source_rows = read_manifest(source_manifest_path)
        processed_path = Path(args.preprocessed) / processed_names[split_name]
        rows = read_manifest(processed_path)
        source_ids = [row["image_id"] for row in source_rows]
        processed_ids = [row["image_id"] for row in rows]
        split_ids[split_name] = set(source_ids)
        processed_rows[split_name] = rows
        if len(source_ids) != expected_count:
            issues.append(
                f"{split_name} has {len(source_ids)} images; "
                f"{expected_count} are required."
            )
        if len(source_ids) != len(set(source_ids)):
            issues.append(f"{split_name} source manifest has duplicate IDs.")
        if processed_ids != source_ids:
            issues.append(
                f"{split_name} preprocessed IDs/order differ from the frozen manifest."
            )
        for row in rows:
            path = Path(row["path"])
            if not path.is_file():
                issues.append(
                    f"{split_name} image {row['image_id']} is missing: {path}"
                )
                continue
            try:
                from PIL import Image

                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    if image.mode != "RGB":
                        issues.append(
                            f"{split_name} image {row['image_id']} is not RGB."
                        )
            except (OSError, ValueError, SyntaxError) as error:
                issues.append(
                    f"{split_name} image {row['image_id']} is corrupt: {error}"
                )
                continue
            if sha256_file(path) != row["sha256"]:
                issues.append(
                    f"{split_name} image {row['image_id']} checksum differs "
                    "from its preprocessed manifest."
                )

    overlaps = {
        "knowledge_base_validation": sorted(
            split_ids["knowledge_base"] & split_ids["validation"]
        ),
        "knowledge_base_test": sorted(
            split_ids["knowledge_base"] & split_ids["test"]
        ),
        "validation_test": sorted(
            split_ids["validation"] & split_ids["test"]
        ),
    }
    if any(overlaps.values()):
        issues.append("Frozen split IDs overlap.")

    caption_counts = {}
    caption_links_valid = True
    for split_name in expected_counts:
        source_name = (
            "train_captions.csv"
            if split_name == "knowledge_base"
            else f"{split_name}_captions.csv"
        )
        processed_name = (
            "train_captions.csv"
            if split_name == "knowledge_base"
            else f"{split_name}_captions.csv"
        )
        source_path = Path(split_manifest["manifest_files"][source_name])
        processed_path = Path(args.preprocessed) / processed_name
        with source_path.open("r", encoding="utf-8", newline="") as file:
            source_captions = list(csv.DictReader(file))
        with processed_path.open("r", encoding="utf-8", newline="") as file:
            captions = list(csv.DictReader(file))
        source_signature = sorted(
            (row["caption_id"], row["image_id"], row["raw_caption"])
            for row in source_captions
        )
        processed_signature = sorted(
            (row["caption_id"], row["image_id"], row["raw_caption"])
            for row in captions
        )
        if source_signature != processed_signature:
            caption_links_valid = False
            issues.append(
                f"{split_name} raw captions or caption mappings differ "
                "from the frozen manifest."
            )
        valid_ids = split_ids[split_name]
        if any(row["image_id"] not in valid_ids for row in captions):
            caption_links_valid = False
            issues.append(f"{split_name} contains captions linked to other splits.")
        caption_counts[split_name] = len(captions)

    metadata_rows = load_json(Path(args.preprocessed) / "metadata.json")
    metadata_ids = [image_id_key(row["image_id"]) for row in metadata_rows]
    expected_all_ids = set().union(*split_ids.values())
    if len(metadata_ids) != len(set(metadata_ids)):
        issues.append("Preprocessed metadata has duplicate image IDs.")
    if set(metadata_ids) != expected_all_ids:
        issues.append("Preprocessed metadata coverage differs from frozen splits.")

    report = {
        "version": 1,
        "passed": not issues,
        "issues": issues,
        "split_counts": {
            split_name: len(split_ids[split_name])
            for split_name in expected_counts
        },
        "processed_image_count": sum(
            len(rows) for rows in processed_rows.values()
        ),
        "processed_images_rgb_and_checksum_verified": not any(
            "not RGB" in issue
            or "checksum differs" in issue
            or "is corrupt" in issue
            or "is missing" in issue
            for issue in issues
        ),
        "caption_counts": caption_counts,
        "caption_links_and_raw_text_verified": caption_links_valid,
        "metadata_count": len(metadata_rows),
        "metadata_coverage_verified": set(metadata_ids) == expected_all_ids,
        "split_id_overlaps": overlaps,
        "test_reference_count": caption_counts["test"],
        "test_reference_count_expected": expected_counts["test"] * 5,
    }
    if report["test_reference_count"] != report["test_reference_count_expected"]:
        issues.append("Test reference count is not exactly 500.")
        report["passed"] = False
    write_json(args.output, report)
    if issues:
        raise ValueError(
            f"Dataset quality checks failed ({len(issues)} issue(s)); "
            f"see {Path(args.output).resolve()}."
        )
    print(f"Dataset quality checks passed; report written to {Path(args.output).resolve()}")


def resolve_split_manifest_path(split_source):
    split_path = Path(split_source)
    if split_path.is_dir():
        for candidate_name in ("splits_v2.json", "split_ids.json"):
            candidate = split_path / candidate_name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"Split directory {split_path} does not contain a supported manifest "
            "(splits_v2.json or split_ids.json)."
        )
    if split_path.is_file():
        if split_path.name in {"splits_v2.json", "split_ids.json"}:
            return split_path
        raise FileNotFoundError(
            f"Unsupported split manifest: {split_path}. Use splits_v2.json or "
            "split_ids.json."
        )
    raise FileNotFoundError(f"Split source does not exist: {split_path}")


def load_split_partition_ids(split_source):
    split_path = resolve_split_manifest_path(split_source)
    split_data = load_json(split_path)
    if not isinstance(split_data, dict):
        raise ValueError(f"Split manifest {split_path} must contain a JSON object.")
    if set(split_data).intersection({"knowledge_base", "validation", "test"}):
        return {
            split_name: {
                image_id_key(image_id)
                for image_id in split_data.get(split_name, [])
            }
            for split_name in ("knowledge_base", "validation", "test")
        }
    if "train" in split_data and "validation" in split_data and "test" in split_data:
        return {
            "knowledge_base": {
                image_id_key(image_id)
                for image_id in split_data.get("train", [])
            },
            "validation": {
                image_id_key(image_id)
                for image_id in split_data.get("validation", [])
            },
            "test": {
                image_id_key(image_id)
                for image_id in split_data.get("test", [])
            },
        }
    if "manifest_files" in split_data:
        manifest_names = {
            "knowledge_base": "knowledge_base_manifest.csv",
            "validation": "validation_manifest.csv",
            "test": "test_manifest.csv",
        }
        manifest_files = split_data.get("manifest_files", {})
        result = {}
        for split_name, file_name in manifest_names.items():
            manifest_entry = manifest_files.get(file_name)
            if manifest_entry:
                manifest_path = Path(manifest_entry)
            else:
                manifest_path = split_path.parent / file_name
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Missing frozen {split_name} manifest: {manifest_path}"
                )
            result[split_name] = {
                row["image_id"] for row in read_manifest(manifest_path)
            }
        return result
    raise ValueError(
        f"Split manifest {split_path} does not contain a supported image ID layout."
    )


def quality_check(args):
    if not args.embeddings:
        if not args.splits or not args.preprocessed:
            raise ValueError(
                "Dataset-only quality checks require --splits and --preprocessed."
            )
        return dataset_quality_check(args)
    if not args.manifest:
        raise ValueError("Embedding quality checks require --manifest.")
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Quality checks require numpy. Install requirements-embeddings.txt."
        ) from error

    records = read_manifest(args.manifest)
    manifest_ids = [image_id_key(row["image_id"]) for row in records]
    data = load_embeddings(args.embeddings)
    image_ids = [image_id_key(value) for value in data["image_ids"].tolist()]
    image_embeddings = np.asarray(data["image_embeddings"], dtype=np.float32)
    text_embeddings = np.asarray(data["text_embeddings"], dtype=np.float32)
    caption_image_ids = [
        image_id_key(value) for value in data["caption_image_ids"].tolist()
    ]
    caption_ids = [str(value) for value in data["caption_ids"].tolist()]
    archive_name = Path(args.embeddings).name.lower()
    is_clip_archive = "clip" in archive_name or "clip-" in archive_name
    is_sbert_archive = "sbert" in archive_name or "sentence" in archive_name
    issues = []

    if len(manifest_ids) != len(set(manifest_ids)):
        issues.append("Manifest contains duplicate image IDs.")
    if len(image_ids) != len(set(image_ids)):
        issues.append("Image embeddings contain duplicate image IDs.")
    if image_embeddings.size:
        if image_embeddings.ndim != 2:
            issues.append("Image embeddings are not a 2D matrix.")
        if is_clip_archive and image_embeddings.shape[1] != 512:
            issues.append(
                f"CLIP image embeddings must have 512 columns; got {image_embeddings.shape}."
            )
        if list(image_ids) != manifest_ids:
            issues.append("Image embedding IDs or ordering differ from the manifest.")
        missing_embeddings = sorted(set(manifest_ids) - set(image_ids))
        unexpected_embeddings = sorted(set(image_ids) - set(manifest_ids))
        if missing_embeddings:
            issues.append(
                f"Missing image embeddings for {len(missing_embeddings)} IDs."
            )
        if unexpected_embeddings:
            issues.append(
                f"Embeddings include {len(unexpected_embeddings)} unexpected IDs."
            )
        if not np.isfinite(image_embeddings).all():
            issues.append("Image embeddings contain NaN or infinite values.")
        image_norms = np.linalg.norm(image_embeddings, axis=1)
        if not np.allclose(image_norms, 1, atol=1e-3):
            issues.append("Image embeddings are not L2-normalized.")
    elif len(image_ids) and list(image_ids) != manifest_ids:
        issues.append("Image embedding IDs or ordering differ from the manifest.")

    if text_embeddings.size:
        if text_embeddings.ndim != 2:
            issues.append("Text embeddings are not a 2D matrix.")
        if is_clip_archive and text_embeddings.shape[1] != 512:
            issues.append(
                "CLIP text embeddings must be 512-dimensional; "
                f"got shape {text_embeddings.shape}."
            )
        if is_sbert_archive and text_embeddings.shape[1] != 384:
            issues.append(
                "Sentence-BERT text embeddings must be 384-dimensional; "
                f"got shape {text_embeddings.shape}."
            )
        if len(caption_image_ids) != len(text_embeddings):
            issues.append("Text embedding count does not match caption image IDs.")
        if len(caption_ids) != len(text_embeddings):
            issues.append("Text embedding count does not match caption IDs.")
        if len(data["captions"]) and len(data["captions"]) != len(text_embeddings):
            issues.append("Text embedding count does not match caption text rows.")
        if len(caption_image_ids) != len(set(zip(caption_image_ids, caption_ids))):
            issues.append("Text embeddings contain duplicate caption IDs.")
        if set(caption_image_ids) - set(manifest_ids):
            issues.append("Text embeddings include parent-image IDs outside the manifest.")
        if not np.isfinite(text_embeddings).all():
            issues.append("Text embeddings contain NaN or infinite values.")
        text_norms = np.linalg.norm(text_embeddings, axis=1)
        if not np.allclose(text_norms, 1, atol=1e-3):
            issues.append("Text embeddings are not L2-normalized.")

    caption_count_mismatch = None
    caption_reference_count_mismatch = None
    if args.captions:
        with Path(args.captions).open("r", encoding="utf-8", newline="") as file:
            caption_rows_input = list(csv.DictReader(file))
        expected = [
            (
                image_id_key(row["image_id"]),
                str(row.get("caption_id", "")),
                row.get("raw_caption", row.get("caption", "")),
                row.get(
                    "cleaned_caption",
                    clean_caption(
                        row.get("raw_caption", row.get("caption", ""))
                    ),
                ),
            )
            for row in caption_rows_input
        ]
        actual = list(
            zip(
                caption_image_ids,
                caption_ids,
                [str(value) for value in data["captions"].tolist()],
                [str(value) for value in data["cleaned_captions"].tolist()],
            )
        )
        if expected and actual and expected != actual:
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
        if args.captions_per_image:
            caption_counts = Counter(
                image_id_key(row["image_id"]) for row in caption_rows_input
            )
            caption_reference_count_mismatch = [
                {"image_id": image_id, "caption_count": caption_counts[image_id]}
                for image_id in manifest_ids
                if caption_counts.get(image_id, 0) != args.captions_per_image
            ]
            if caption_reference_count_mismatch:
                issues.append(
                    "Caption reference count differs from the expected "
                    f"{args.captions_per_image} for "
                    f"{len(caption_reference_count_mismatch)} image IDs."
                )

    split_audit = {}
    if args.splits:
        raw_split_data = load_json(resolve_split_manifest_path(args.splits))
        split_ids = load_split_partition_ids(args.splits)
        knowledge_base_ids = set(split_ids["knowledge_base"])
        validation_ids = set(split_ids["validation"])
        test_ids = set(split_ids["test"])
        legacy_overlap_is_allowed = (
            "train" in raw_split_data
            and validation_ids.issubset(knowledge_base_ids)
        )
        split_audit = {
            "knowledge_base_validation_disjoint": (
                not (knowledge_base_ids & validation_ids)
                or legacy_overlap_is_allowed
            ),
            "knowledge_base_test_disjoint": not (knowledge_base_ids & test_ids),
            "validation_test_disjoint": not (validation_ids & test_ids),
            "validation_is_subset_of_knowledge_base": validation_ids.issubset(
                knowledge_base_ids
            ),
            "knowledge_base_and_validation_cover_split": (
                knowledge_base_ids | validation_ids
            ) == knowledge_base_ids,
        }
        if not all(split_audit.values()) and not legacy_overlap_is_allowed:
            issues.append("Split ID audit found overlapping image IDs.")
        if set(image_ids).intersection(test_ids):
            issues.append("Test images are present in retrieval embeddings.")
        if set(caption_image_ids).intersection(test_ids):
            issues.append("Test captions are present in retrieval embeddings.")

    metadata_count = None
    missing_metadata_ids = []
    if args.metadata:
        metadata_rows = load_json(args.metadata)
        if not isinstance(metadata_rows, list):
            raise ValueError("Metadata JSON must contain a list of image records.")
        metadata_ids = [image_id_key(row["image_id"]) for row in metadata_rows]
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
        "image_embedding_count": len(image_ids),
        "text_embedding_count": len(text_embeddings),
        "embedding_dimension": (
            int(image_embeddings.shape[1]) if image_embeddings.size else None
        ),
        "text_embedding_dimension": (
            int(text_embeddings.shape[1]) if text_embeddings.size else None
        ),
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


def create_release_manifest(args):
    config, root = load_config(args.config)
    split_path = Path(args.splits).resolve()
    split_manifest = load_json(split_path)
    if not isinstance(split_manifest.get("counts"), dict):
        raise ValueError("Split manifest does not contain dataset counts.")

    def relative_path(path):
        try:
            return str(Path(path).resolve().relative_to(root))
        except ValueError:
            return str(Path(path).resolve())

    artifact_paths = {split_path}
    for path in split_manifest.get("manifest_files", {}).values():
        artifact_paths.add(Path(path))
    for directory in (
        root / "reports" / "phase1",
        root / "01_splits",
    ):
        if directory.is_dir():
            artifact_paths.update(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and path.suffix.lower() not in {".jpg", ".jpeg"}
                and "history" not in path.parts
            )
    for path in (
        root / "REPORT.md",
        root / "README.md",
        root / "scripts" / "phase1.py",
    ):
        if path.is_file():
            artifact_paths.add(path)
    artifacts = {}
    for path in sorted(artifact_paths, key=str):
        if path.is_file():
            artifacts[relative_path(path)] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

    git_commit = None
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if git_result.returncode == 0:
        git_commit = git_result.stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    config_path = Path(args.config).resolve()
    counts = split_manifest["counts"]
    processed_metadata = root / "reports" / "phase1" / "preprocessed" / "metadata.json"
    preprocessing_complete = processed_metadata.is_file()
    payload = {
        "version": "0.2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_worktree_dirty": (
            bool(git_status.stdout.strip()) if git_status.returncode == 0 else None
        ),
        "configuration": {
            "path": relative_path(config_path),
            "sha256": sha256_file(config_path),
        },
        "split_manifest": {
            "path": relative_path(split_path),
            "version": split_manifest["manifest_version"],
            "sha256": sha256_file(split_path),
        },
        "dataset_counts": {
            "candidate_pool": counts.get("candidate_pool_valid_unique"),
            "knowledge_base_images": counts.get("knowledge_base_images"),
            "validation_images": counts.get("validation_images"),
            "test_images": counts.get("test_images"),
            "reserve_images": counts.get("reserve_images"),
            "knowledge_base_captions": counts.get("knowledge_base_captions"),
            "validation_captions": counts.get("validation_captions"),
            "test_captions": counts.get("test_captions"),
        },
        "models": {
            "clip": {
                "name": config.get("embedding", {}).get("clip_model"),
                "version": None,
                "dimension": 512,
                "executed": False,
            },
            "sentence_bert": {
                "name": config.get("embedding", {}).get(
                    "sentence_transformer_model"
                ),
                "version": None,
                "dimension": 384,
                "executed": False,
            },
        },
        "indexes": {
            "configured_faiss_type": config.get("index", {}).get("type"),
            "built": False,
            "vector_counts": None,
        },
        "quality_control": {
            "split_leakage_audit": split_manifest.get("leakage_audit"),
            "caption_integrity": split_manifest.get("caption_integrity"),
            "status": "partial",
            "preprocessing_metadata_present": preprocessing_complete,
            "embedding_and_index_checks": "not executed",
        },
        "status": {
            "complete": False,
            "state": "incomplete",
            "completed": [
                "COCO source audit",
                "candidate-pool selection and file validation",
                "versioned disjoint split creation",
                "caption-reference validation",
                "cross-split exact and perceptual duplicate checks",
                "knowledge-base exploratory analysis",
            ]
            + (["RGB/EXIF image preprocessing"] if preprocessing_complete else []),
            "not_executed": [
                "CLIP and Sentence-BERT embedding generation",
                "FAISS index construction",
                "end-to-end retrieval validation",
                "embedding and index quality checks",
                "semantic retrieval evaluation",
            ],
            "optional_not_supported_by_source": [
                "Verified scene labels, visual attributes, and semantic spatial relations "
                "are not supplied by standard COCO annotations."
            ],
        },
        "artifacts": artifacts,
    }
    write_json(args.output, payload)
    print(f"Release manifest written to {Path(args.output).resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    audit_parser = commands.add_parser(
        "audit", help="Audit COCO annotation IDs, caption coverage, and files."
    )
    audit_parser.add_argument("--config", default="configs/phase1.yaml")
    audit_parser.add_argument("--instances")
    audit_parser.add_argument("--captions")
    audit_parser.add_argument("--images")
    audit_parser.add_argument("--output")
    audit_parser.set_defaults(func=audit_dataset)

    explore_parser = commands.add_parser(
        "explore", help="Create dataset statistics and visualizations."
    )
    explore_parser.add_argument("--instances", required=True)
    explore_parser.add_argument("--captions", required=True)
    explore_parser.add_argument("--output", required=True)
    explore_parser.add_argument(
        "--manifest",
        help="Optional frozen knowledge_base_manifest.csv to limit EDA to KB IDs.",
    )
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

    select_parser = commands.add_parser(
        "select",
        help="Sample and audit a deterministic train2017 candidate pool.",
    )
    select_parser.add_argument("--config", default="configs/phase1.yaml")
    select_parser.add_argument("--instances")
    select_parser.add_argument("--captions")
    select_parser.add_argument("--images")
    select_parser.add_argument("--output")
    select_parser.add_argument("--candidate-count", type=int)
    select_parser.add_argument("--knowledge-base-count", type=int)
    select_parser.add_argument("--validation-count", type=int)
    select_parser.add_argument("--captions-per-image", type=int)
    select_parser.add_argument("--seed", type=int)
    select_parser.set_defaults(func=select_candidates)

    split_parser = commands.add_parser(
        "split", help="Freeze disjoint train2017 and val2017 dataset manifests."
    )
    split_parser.add_argument("--config", default="configs/phase1.yaml")
    split_parser.add_argument("--manifest", help="Audited candidate_manifest.csv.")
    split_parser.add_argument("--instances", help="COCO train2017 instances JSON.")
    split_parser.add_argument("--captions", help="COCO train2017 captions JSON.")
    split_parser.add_argument("--test-instances", help="COCO val2017 instances JSON.")
    split_parser.add_argument("--test-captions", help="COCO val2017 captions JSON.")
    split_parser.add_argument("--test-images", help="COCO val2017 image directory.")
    split_parser.add_argument("--output")
    split_parser.add_argument("--knowledge-base-count", type=int)
    split_parser.add_argument("--validation-count", type=int)
    split_parser.add_argument("--test-count", type=int)
    split_parser.add_argument("--captions-per-image", type=int)
    split_parser.add_argument("--seed", type=int)
    split_parser.add_argument("--near-duplicate-threshold", type=float)
    split_parser.add_argument(
        "--force-new-splits",
        action="store_true",
        help="Archive the current frozen split manifest and explicitly regenerate.",
    )
    split_parser.set_defaults(func=freeze_splits)

    preprocess_parser = commands.add_parser(
        "preprocess",
        help="Create RGB images, isolated caption files, and linked metadata.",
    )
    preprocess_parser.add_argument("--config", default="configs/phase1.yaml")
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
        "embed", help="Build normalized CLIP or Sentence-BERT embeddings."
    )
    embed_parser.add_argument("--config", default="configs/phase1.yaml")
    embed_parser.add_argument("--manifest", required=True, help="Use the frozen knowledge-base manifest.")
    embed_parser.add_argument(
        "--captions",
        help="Optional COCO captions JSON or prepared caption CSV.",
    )
    embed_parser.add_argument("--output", required=True, help="Output .npz file.")
    embed_parser.add_argument(
        "--encoder", choices=("clip", "sbert"), default="clip"
    )
    embed_parser.add_argument("--model")
    embed_parser.add_argument("--batch-size", type=int, default=32)
    embed_parser.add_argument("--device", help="Defaults to CUDA when available, otherwise CPU.")
    embed_parser.set_defaults(func=embed)

    index_parser = commands.add_parser(
        "build-index", help="Build a reproducible exact cosine vector index."
    )
    index_parser.add_argument("--config", default="configs/phase1.yaml")
    index_parser.add_argument("--embeddings", required=True)
    index_parser.add_argument("--metadata", required=True)
    index_parser.add_argument("--output", required=True, help="Output .npz file.")
    index_parser.set_defaults(func=build_index)

    faiss_parser = commands.add_parser(
        "build-faiss-index",
        help="Build separate CLIP image, CLIP caption, and Sentence-BERT indexes.",
    )
    faiss_parser.add_argument("--config", default="configs/phase1.yaml")
    faiss_parser.add_argument("--clip-embeddings", required=True)
    faiss_parser.add_argument("--sbert-embeddings", required=True)
    faiss_parser.add_argument("--metadata", required=True)
    faiss_parser.add_argument("--output", required=True, help="Index output directory.")
    faiss_parser.add_argument("--overwrite", action="store_true")
    faiss_parser.set_defaults(func=build_faiss_indexes)

    encode_parser = commands.add_parser(
        "encode-query",
        help="Encode an image or text query in its matching retrieval space.",
    )
    encode_parser.add_argument("--config", default="configs/phase1.yaml")
    encode_parser.add_argument(
        "--encoder",
        required=True,
        choices=("clip-image", "clip-text", "sbert-text"),
    )
    encode_parser.add_argument("--image")
    encode_parser.add_argument("--text")
    encode_parser.add_argument("--model")
    encode_parser.add_argument("--device")
    encode_parser.add_argument("--output", required=True, help="Output .npy vector.")
    encode_parser.set_defaults(func=encode_query)

    retrieve_parser = commands.add_parser(
        "retrieve", help="Query a saved FAISS index and apply image-ID exclusions."
    )
    retrieve_parser.add_argument("--config", default="configs/phase1.yaml")
    retrieve_parser.add_argument("--index", required=True)
    retrieve_parser.add_argument("--mapping", required=True)
    retrieve_parser.add_argument("--query-vector", required=True)
    retrieve_parser.add_argument("--top-k", type=int, default=5)
    retrieve_parser.add_argument(
        "--exclude-image-id", action="append", default=[]
    )
    retrieve_parser.add_argument("--output", required=True)
    retrieve_parser.set_defaults(func=retrieve_faiss)

    quality_parser = commands.add_parser(
        "quality-check", help="Audit embeddings, captions, and split leakage."
    )
    quality_parser.add_argument("--config", default="configs/phase1.yaml")
    quality_parser.add_argument("--manifest")
    quality_parser.add_argument("--embeddings")
    quality_parser.add_argument(
        "--captions", help="Prepared caption CSV corresponding to the embeddings."
    )
    quality_parser.add_argument(
        "--splits",
        help="Split output directory or splits_v2.json for dataset checks.",
    )
    quality_parser.add_argument(
        "--preprocessed",
        help="Preprocessed output directory for dataset-only quality checks.",
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

    release_parser = commands.add_parser(
        "release-manifest",
        help="Record release provenance, artifact hashes, and execution status.",
    )
    release_parser.add_argument("--config", default="configs/phase1.yaml")
    release_parser.add_argument("--splits", default="01_splits/splits_v2.json")
    release_parser.add_argument(
        "--output", default="release_manifest.json"
    )
    release_parser.set_defaults(func=create_release_manifest)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {
        "audit",
        "select",
        "split",
        "preprocess",
        "embed",
        "build-index",
        "build-faiss-index",
        "encode-query",
        "retrieve",
        "quality-check",
        "release-manifest",
    }:
        try:
            apply_config(args)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        required_by_command = {
            "audit": ("instances", "captions", "output"),
            "select": (
                "instances",
                "captions",
                "images",
                "output",
                "candidate_count",
                "knowledge_base_count",
                "validation_count",
                "captions_per_image",
                "seed",
            ),
            "split": (
                "manifest",
                "instances",
                "captions",
                "test_instances",
                "test_captions",
                "test_images",
                "output",
                "knowledge_base_count",
                "validation_count",
                "test_count",
                "captions_per_image",
                "seed",
                "near_duplicate_threshold",
            ),
            "preprocess": (
                "splits",
                "instances",
                "captions",
                "output",
            ),
            "embed": ("manifest", "output"),
            "build-index": ("embeddings", "metadata", "output"),
            "build-faiss-index": (
                "clip_embeddings",
                "sbert_embeddings",
                "metadata",
                "output",
            ),
            "quality-check": ("output",),
            "release-manifest": ("splits", "output"),
        }
        missing = [
            name for name in required_by_command.get(args.command, ())
            if getattr(args, name, None) is None
        ]
        if missing:
            parser.error(
                f"{args.command} requires configuration or CLI values for: "
                f"{', '.join('--' + name.replace('_', '-') for name in missing)}"
            )
    if getattr(args, "batch_size", 1) <= 0:
        parser.error("--batch-size must be greater than zero.")
    if getattr(args, "top_k", 1) <= 0:
        parser.error("--top-k must be greater than zero.")
    if args.command == "encode-query":
        if args.encoder == "clip-image" and not args.image:
            parser.error("--encoder clip-image requires --image.")
        if args.encoder in {"clip-text", "sbert-text"} and not args.text:
            parser.error(f"--encoder {args.encoder} requires --text.")
    for argument in ("max_size", "captions_per_image"):
        if getattr(args, argument, 0) < 0:
            parser.error(f"--{argument.replace('_', '-')} must not be negative.")
    if (
        getattr(args, "confidence_threshold", None) is not None
        and not 0 <= args.confidence_threshold <= 1
    ):
        parser.error("--confidence-threshold must be between zero and one.")
    if (
        getattr(args, "near_duplicate_threshold", None) is not None
        and not 0 <= args.near_duplicate_threshold <= 1
    ):
        parser.error("--near-duplicate-threshold must be between zero and one.")
    for argument in (
        "expected_count",
        "candidate_count",
        "knowledge_base_count",
        "validation_count",
        "test_count",
    ):
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
