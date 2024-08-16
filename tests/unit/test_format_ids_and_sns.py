import unittest
from unittest.mock import patch, mock_open
from format_ids_and_sns import overwrite, transform

"PYTHONPATH=~/Test_Equipment/crispy-garland /bin/python3 ~/Test_Equipment/crispy-garland/tests/unit/test_format_ids_and_sns.py"


class TestOverwrite(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open, read_data="1\n2\n3\n4\n5\n6\n")
    def test_transform_and_overwrite(self, mock_file):
        input_file = "mock_input.txt"
        expected_output = "1\t2\n3\t4\n5\t6\n"
        overwrite(input_file)
        mock_file.assert_called_with(input_file, "w")
        handle = mock_file()
        handle.write.assert_called_once_with(expected_output)


class TestTransform(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(transform([]), [])

    def test_single_line(self):
        self.assertEqual(transform(["line1"]), ["line1"])

    def test_odd_number_of_lines(self):
        self.assertEqual(
            transform(["line1", "line2", "line3"]), ["line1\tline2", "line3"]
        )

    def test_even_number_of_lines(self):
        self.assertEqual(
            transform(["line1", "line2", "line3", "line4"]),
            ["line1\tline2", "line3\tline4"],
        )

    def test_long_list(self):
        self.assertEqual(
            transform(["line1", "line2", "line3", "line4", "line5", "line6"]),
            ["line1\tline2", "line3\tline4", "line5\tline6"],
        )

    def test_empty_strings(self):
        self.assertEqual(transform(["", ""]), ["\t"])

    def test_different_lengths(self):
        self.assertEqual(transform(["line1", "longer line2"]), ["line1\tlonger line2"])


if __name__ == "__main__":
    unittest.main()
