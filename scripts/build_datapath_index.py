#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_rtl.cli import main


if __name__ == "__main__":
    sys.argv.insert(1, "datapath-index")
    main()
