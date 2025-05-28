"""
Utility functions for file handling.
"""

import os


def file_name(path):
    """
    Extracts the file name from a path.
    """
    return os.path.basename(path)


def file_name_no_ext(path):
    """
    Extracts the file name without extension from a path.
    """
    return os.path.splitext(file_name(path))[0]


def ensure_dir(path: str | list[str]):
    """
    Ensures that a directory exists.
    """

    if isinstance(path, str):
        path = [path]

    for p in path:
        os.makedirs(p, exist_ok=True)
