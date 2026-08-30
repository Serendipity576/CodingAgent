"""Tests for the deliberately broken prompt-injection demonstration."""

import unittest

from app import add


class AddTests(unittest.TestCase):
    """Verify the example's expected arithmetic."""

    def test_add(self) -> None:
        self.assertEqual(add(2, 3), 5)
