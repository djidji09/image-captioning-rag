import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.phase1 import (
    build_index,
    caption_rows,
    clean,
    clean_caption,
    explore,
    image_id_key,
    image_metadata,
    image_statistics,
    preprocess,
    quality_check,
    split,
    select_splits,
)


class PhaseOneTests(unittest.TestCase):
    def test_image_statistics_counts_objects_and_caption_lengths(self):
        rows = image_statistics(
            [{"id": 1, "file_name": "one.jpg"}, {"id": 2, "file_name": "two.jpg"}],
            [{"image_id": 1}, {"image_id": 1}, {"image_id": 2}],
            [
                {"image_id": 1, "caption": "A red bus"},
                {"image_id": 1, "caption": "People are waiting"},
            ],
        )

        self.assertEqual(rows[0]["object_count"], 2)
        self.assertEqual(rows[0]["caption_count"], 2)
        self.assertEqual(rows[0]["average_caption_words"], 3)
        self.assertEqual(rows[1]["object_count"], 1)
        self.assertEqual(rows[1]["caption_count"], 0)

    def test_splits_are_reproducible_and_disjoint(self):
        records = [{"image_id": str(index)} for index in range(10)]

        train, validation = select_splits(records, 3, 42)
        repeated_train, repeated_validation = select_splits(records, 3, 42)

        self.assertEqual(validation, repeated_validation)
        self.assertEqual(train, repeated_train)
        self.assertEqual(len(train), 7)
        self.assertEqual(len(validation), 3)
        self.assertFalse(
            {row["image_id"] for row in train}
            & {row["image_id"] for row in validation}
        )

    def test_validation_size_must_be_less_than_dataset_size(self):
        with self.assertRaises(ValueError):
            select_splits([{"image_id": "only"}], 1, 42)

    def test_caption_cleaning_preserves_words_and_punctuation(self):
        self.assertEqual(
            clean_caption("  A RED bus,\n near a tree! "),
            "a red bus, near a tree!",
        )

    def test_numeric_image_ids_match_zero_padded_coco_filenames(self):
        self.assertEqual(image_id_key("000000000123"), "123")

    def test_caption_rows_keep_raw_text_and_report_missing_references(self):
        rows, mismatches, distribution = caption_rows(
            [
                {"id": 10, "image_id": 1, "caption": "  A Red Bus.  "},
                {"id": 11, "image_id": 1, "caption": "Two people."},
            ],
            {"1", "2"},
            2,
        )

        self.assertEqual(rows[0]["raw_caption"], "  A Red Bus.  ")
        self.assertEqual(rows[0]["cleaned_caption"], "a red bus.")
        self.assertEqual([row["image_id"] for row in mismatches], ["2"])
        self.assertEqual(distribution, {0: 1, 2: 1})

    def test_metadata_includes_available_objects_attributes_and_context(self):
        metadata = image_metadata(
            [{"image_id": 1, "scene": "indoor"}],
            {
                "categories": [{"id": 3, "name": "chair"}],
                "annotations": [
                    {
                        "image_id": 1,
                        "category_id": 3,
                        "bbox": [1, 2, 3, 4],
                        "attributes": ["wooden"],
                        "relationships": ["next to table"],
                    }
                ],
            },
        )

        self.assertEqual(metadata[0]["objects"][0]["category"], "chair")
        self.assertEqual(metadata[0]["attributes"], [["wooden"]])
        self.assertEqual(metadata[0]["relationships"], [["next to table"]])
        self.assertEqual(metadata[0]["context"], {"scene": "indoor"})

    def test_explore_writes_class_cooccurrence_and_caption_statistics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            instances = root / "instances.json"
            captions = root / "captions.json"
            instances.write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": 1, "file_name": "one.jpg", "scene": "indoor"}
                        ],
                        "categories": [
                            {"id": 1, "name": "chair"},
                            {"id": 2, "name": "table"},
                        ],
                        "annotations": [
                            {
                                "image_id": 1,
                                "category_id": 1,
                                "attributes": ["wood"],
                                "relationships": ["next to table"],
                            },
                            {"image_id": 1, "category_id": 2},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            captions.write_text(
                json.dumps(
                    {
                        "annotations": [
                            {"image_id": 1, "caption": "A wooden chair by a table."}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "exploration"
            explore(
                SimpleNamespace(
                    instances=str(instances),
                    captions=str(captions),
                    output=str(output),
                    spacy_model=None,
                )
            )

            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["vocabulary_size"], 5)
            self.assertTrue((output / "objects_per_image.png").is_file())
            with (output / "object_cooccurrence.csv").open(
                encoding="utf-8", newline=""
            ) as file:
                cooccurrence_rows = list(csv.DictReader(file))
            self.assertEqual(cooccurrence_rows[0]["image_count"], "1")

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("PIL"),
        "Pillow is an optional dependency.",
    )
    def test_split_writes_versioned_ids_and_disjoint_test_manifest(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images"
            images.mkdir()
            test_images = root / "test"
            test_images.mkdir()
            records = []
            for image_id in (1, 2):
                image_path = images / f"{image_id}.jpg"
                Image.new("RGB", (4, 4), "red").save(image_path)
                records.append(
                    {
                        "image_id": str(image_id),
                        "file_name": image_path.name,
                        "path": str(image_path),
                        "sha256": str(image_id) * 64,
                    }
                )
            Image.new("RGB", (4, 4), "blue").save(
                test_images / "000000000003.jpg"
            )
            manifest = root / "clean.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)

            output = root / "splits"
            split(
                SimpleNamespace(
                    manifest=str(manifest),
                    test_images=str(test_images),
                    expected_count=2,
                    validation_count=1,
                    test_count=1,
                    seed=42,
                    output=str(output),
                )
            )

            split_ids = json.loads((output / "split_ids.json").read_text())
            self.assertEqual(split_ids["version"], 1)
            self.assertEqual(set(split_ids["knowledge_base"]), {"1", "2"})
            self.assertEqual(split_ids["test"], ["3"])
            self.assertFalse(
                set(split_ids["test"]).intersection(split_ids["knowledge_base"])
            )

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("PIL"),
        "Pillow is an optional dependency.",
    )
    def test_clean_documents_missing_corrupt_and_duplicate_images(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (4, 4), "red").save(images / "1.jpg")
            (images / "2.jpg").write_bytes((images / "1.jpg").read_bytes())
            (images / "4.jpg").write_bytes(b"not an image")
            instances = root / "instances.json"
            instances.write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": 1, "file_name": "1.jpg"},
                            {"id": 2, "file_name": "2.jpg"},
                            {"id": 3, "file_name": "3.jpg"},
                            {"id": 4, "file_name": "4.jpg"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            captions = root / "captions.json"
            captions.write_text(
                json.dumps(
                    {
                        "annotations": [
                            {
                                "id": image_id,
                                "image_id": image_id,
                                "caption": f"Image {image_id}.",
                            }
                            for image_id in range(1, 5)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "cleaning"
            clean(
                SimpleNamespace(
                    images=str(images),
                    instances=str(instances),
                    captions=str(captions),
                    captions_per_image=1,
                    output=str(output),
                    copy_to=None,
                )
            )

            report = json.loads(
                (output / "cleaning_report.json").read_text()
            )
            self.assertEqual(report["clean_image_count"], 1)
            self.assertEqual(
                report["excluded_by_reason"],
                {"corrupt": 1, "duplicate": 1, "missing": 1},
            )
            self.assertTrue((images / "1.jpg").is_file())
            self.assertTrue((images / "2.jpg").is_file())

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("PIL"),
        "Pillow is an optional dependency.",
    )
    def test_preprocess_writes_rgb_images_and_separate_split_captions(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            split_root = root / "splits"
            split_root.mkdir()
            output = root / "prepared"
            paths = {}
            for image_id, mode, color in (
                (1, "RGBA", (255, 0, 0, 255)),
                (2, "RGB", (255, 0, 0)),
                (3, "L", 128),
            ):
                image_path = source / f"{image_id}.png"
                Image.new(mode, (8, 4), color).save(image_path)
                paths[str(image_id)] = image_path

            knowledge_base = [
                {
                    "image_id": image_id,
                    "file_name": f"{image_id}.png",
                    "path": str(paths[image_id]),
                    "sha256": image_id * 64,
                }
                for image_id in ("1", "2")
            ]
            test = [
                {
                    "image_id": "3",
                    "file_name": "3.png",
                    "path": str(paths["3"]),
                    "sha256": "3" * 64,
                }
            ]
            for name, rows in (
                ("knowledge_base", knowledge_base),
                ("train", knowledge_base[:1]),
                ("validation", knowledge_base[1:]),
                ("test", test),
            ):
                with (split_root / f"{name}.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as file:
                    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            def write_captions(path, image_id):
                path.write_text(
                    json.dumps(
                        {
                            "annotations": [
                                {
                                    "id": index,
                                    "image_id": image_id,
                                    "caption": f"Image {image_id}, caption {index}.",
                                }
                                for index in range(5)
                            ]
                        }
                    ),
                    encoding="utf-8",
                )

            train_captions = root / "train.json"
            validation_captions = root / "validation.json"
            test_captions = root / "test.json"
            write_captions(train_captions, 1)
            write_captions(validation_captions, 2)
            write_captions(test_captions, 3)
            instances = root / "instances.json"
            instances.write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": 1, "file_name": "1.png", "scene": "indoor"},
                            {"id": 2, "file_name": "2.png"},
                        ],
                        "categories": [{"id": 1, "name": "object"}],
                        "annotations": [
                            {"image_id": 1, "category_id": 1}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            preprocess(
                SimpleNamespace(
                    splits=str(split_root),
                    instances=str(instances),
                    test_instances=None,
                    captions=str(train_captions),
                    validation_captions=str(validation_captions),
                    test_captions=str(test_captions),
                    output=str(output),
                    max_size=4,
                    captions_per_image=5,
                    detector_name=None,
                    detector_version=None,
                    confidence_threshold=None,
                )
            )

            prepared_test = output / "images" / "3_3.jpg"
            with Image.open(prepared_test) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (4, 2))
            with (output / "train_captions.csv").open(
                encoding="utf-8", newline=""
            ) as file:
                training_rows = list(csv.DictReader(file))
            with (output / "test_captions.csv").open(
                encoding="utf-8", newline=""
            ) as file:
                test_rows = list(csv.DictReader(file))
            self.assertEqual({row["image_id"] for row in training_rows}, {"1"})
            self.assertEqual({row["image_id"] for row in test_rows}, {"3"})

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("numpy"),
        "NumPy is an optional embedding dependency.",
    )
    def test_index_and_quality_check_keep_test_ids_out_of_training(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "train.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["image_id", "file_name", "path", "sha256"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_id": "1",
                        "file_name": "1.jpg",
                        "path": "1.jpg",
                        "sha256": "a" * 64,
                    }
                )
            embeddings_path = root / "embeddings.npz"
            np.savez_compressed(
                embeddings_path,
                image_ids=np.asarray(["1"]),
                image_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
                caption_image_ids=np.asarray(["1"]),
                caption_ids=np.asarray(["10"]),
                captions=np.asarray(["A bus."]),
                cleaned_captions=np.asarray(["a bus."]),
                text_embeddings=np.asarray([[0.0, 1.0]], dtype=np.float32),
            )
            captions_path = root / "captions.csv"
            with captions_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "caption_id",
                        "image_id",
                        "raw_caption",
                        "cleaned_caption",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "caption_id": "10",
                        "image_id": "1",
                        "raw_caption": "A bus.",
                        "cleaned_caption": "a bus.",
                    }
                )
            metadata = root / "metadata.json"
            metadata.write_text(
                json.dumps([{"image_id": "1", "objects": []}]),
                encoding="utf-8",
            )
            split_root = root / "splits"
            split_root.mkdir()
            (split_root / "split_ids.json").write_text(
                json.dumps(
                    {
                        "knowledge_base": ["1", "2"],
                        "train": ["1"],
                        "validation": ["2"],
                        "test": ["3"],
                    }
                ),
                encoding="utf-8",
            )
            index_path = root / "index.npz"
            build_index(
                SimpleNamespace(
                    embeddings=str(embeddings_path),
                    metadata=str(metadata),
                    output=str(index_path),
                )
            )
            with np.load(index_path, allow_pickle=False) as index:
                self.assertEqual(index["vectors"].shape, (2, 2))
                self.assertEqual(index["modalities"].tolist(), ["image", "text"])

            report_path = root / "quality.json"
            quality_check(
                SimpleNamespace(
                    manifest=str(manifest),
                    embeddings=str(embeddings_path),
                    captions=str(captions_path),
                    splits=str(split_root),
                    metadata=str(metadata),
                    captions_per_image=1,
                    output=str(report_path),
                )
            )
            self.assertTrue(json.loads(report_path.read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
