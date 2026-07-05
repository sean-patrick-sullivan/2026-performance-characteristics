"""
I/O printing tools for displaying results

This module provides a simple print function for displaying test results.
"""

from typing import Any

# --- Print to TTY utilities ---

def print_section(title: str, data: str | None = None): # type: ignore
    """
    Utility for sectioning output

    Parameters
    ----------
    title : str
        Section title to display
    data : str
        Any simple data to be outptu
    
    Returns
    -------
    None
    """

    # Decorative elements
    title_line = "─" * 72

    # Print heading
    print()
    print(f"{title_line}")
    print(f"{title.center(len(title_line))}")
    print(f"{title_line}")
    print()

    # Print body
    if data is not None:
        print(f"{data}")
        print()

    return data