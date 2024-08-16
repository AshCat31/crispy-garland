import unittest
from unittest.mock import patch


class DM:
    def __init__(self) -> None:
        self.state = "state1"
        self.state_handlers = {
            "state1": self.handle_state_1,
            "state3": self.handle_state_3,
            "state2": self.handle_state_2,
        }

    def mainloop(self):
        while True:
            if self.state in self.state_handlers.keys():
                self.state_handlers[self.state]()
            else:
                raise ValueError("Invalid")

    def handle_state_1(self):
        self.state = "state2"

    def handle_state_2(self):
        self.state = "state3"

    def handle_state_3(self):
        self.state = "state1"


class TestDM(unittest.TestCase):
    def test_mainloop_state_transitions(self):
        dm = DM()
        with patch.object(
            dm,
            "state_handlers",
            {
                "state1": lambda: (
                    self.assertEqual(dm.state, "state1"),
                    setattr(dm, "state", "state2"),
                ),
                "state2": lambda: (
                    self.assertEqual(dm.state, "state2"),
                    setattr(dm, "state", "state3"),
                ),
                "state3": lambda: (
                    self.assertEqual(dm.state, "state3"),
                    setattr(dm, "state", "state4"),
                ),
            },
        ):
            with self.assertRaises(ValueError):
                dm.mainloop()

        self.assertEqual(dm.state, "state4")


if __name__ == "__main__":
    unittest.main()
