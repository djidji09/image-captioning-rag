import unittest

from scripts.phase1 import image_statistics, select_splits


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


if __name__ == "__main__":
    unittest.main()
