#!/usr/bin/env python
import sys
import os
import logging
logging.getLogger("chromadb").setLevel(logging.ERROR)


# Insert the project root into the module search path.
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_dir)

# Import and run the main chat loop.
from core import chat
if __name__ == "__main__":
    chat.main()
