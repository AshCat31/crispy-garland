import unittest
from unittest.mock import Mock

from QA_Check_Auto import JSONChecker

class TestJSONChecker(unittest.TestCase):
    
    def setUp(self):
        self.logger = Mock()
        self.s3client = Mock()
        self.bucket_name = "test_bucket"
        self.values = ["10001", "SN123"]
        self.checker = JSONChecker(self.values, self.s3client, self.bucket_name, self.logger)
        # print('test')
    
    def test_check_js_matches_equal(self):
        self.assertTrue(self.checker.check_js_matches("mosaic", "mosaic", "device_type"))
        self.logger.error.assert_not_called()  # Assert that logger.error was not called
    
    def test_check_js_matches_not_equal(self):
        self.assertFalse(self.checker.check_js_matches("mosaic", "hydra", "device_type"))
        self.logger.error.assert_called_once_with("device_type in data.json is not the same as input.")
    
    def test_get_device_type_known_id(self):
        device_type = self.checker.get_device_type("100001")
        self.assertEqual(device_type, "mosaic")
    
    def test_get_device_type_unknown_id(self):
        device_type = self.checker.get_device_type("XYZ123")
        self.assertEqual(device_type, "unknown")
    
    def test_get_device_type_strip_whitespace(self):
        device_type = self.checker.get_device_type(" E66123 ")
        self.assertEqual(device_type, "hydra")

    def test_get_serial_number_with_serial(self):
        serial_number = self.checker.get_serial_number(["100", "SN123"])
        self.assertEqual(serial_number, "SN123")
        self.logger.warning.assert_not_called()  # Assert that logger.warning was not called
    
    def test_get_serial_number_without_serial(self):
        serial_number = self.checker.get_serial_number(["100"])
        self.assertEqual(serial_number, "none")
        self.logger.warning.assert_called_once_with("SN not found.")


if __name__ == '__main__':
    unittest.main()
