# setup.py

import sys
import os

# Add the parent directory and Stone-Soup to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # The root directory
sys.path.append(os.path.join(parent_dir, 'Stone-Soup'))