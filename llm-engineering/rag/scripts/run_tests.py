"""
Run the full test suite.

Usage:
    python scripts/run_tests.py            # all tests
    python scripts/run_tests.py -v         # verbose
    python scripts/run_tests.py -k chunker # filter by name
"""
import sys
import unittest
from pathlib import Path

# Ensure the installed package is on the path when running without pip install -e
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

TESTS_DIR = Path(__file__).parent.parent / "tests"


def main():
    loader = unittest.TestLoader()
    suite = loader.discover(str(TESTS_DIR), pattern="test_*.py")

    verbosity = 2 if "-v" in sys.argv or "--verbose" in sys.argv else 1

    # Support -k <pattern> filtering
    if "-k" in sys.argv:
        idx = sys.argv.index("-k")
        pattern = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ""
        if pattern:
            def _collect(s, out):
                for item in s:
                    if isinstance(item, unittest.TestSuite):
                        _collect(item, out)
                    elif pattern.lower() in str(item).lower():
                        out.addTest(item)

            filtered = unittest.TestSuite()
            _collect(suite, filtered)
            suite = filtered

    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
