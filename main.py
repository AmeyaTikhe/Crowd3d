"""
main.py -- Root entry point for Crowd3D pipeline.

Runs the complete transparent, non-DL multi-camera crowd density estimation system:
  Geometry -> Detection -> Reconstruction -> Counting -> Evaluation

Usage:
  python main.py --frame 636
  python main.py --frames 636:650:2
  python main.py --preset train_compartment
  python main.py --help
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crowd3d.main import main

if __name__ == "__main__":
    main()
