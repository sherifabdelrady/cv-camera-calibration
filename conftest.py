# conftest.py — pytest configuration for calibvision tests
import sys
from pathlib import Path

# Ensure src/ is always on the path when running pytest from any directory
sys.path.insert(0, str(Path(__file__).parent / "src"))
