"""Common utilities for binning scaleout scripts."""

import glob
import os


def find_missing_file(working_dir, required=False):
    """Find the highest-numbered commands_missing_N.txt in working_dir.
    
    Args:
        working_dir: Directory to search for commands_missing_N.txt files
        required: If True, raise FileNotFoundError if no file found
        
    Returns:
        Filename (relative path) of the highest-numbered commands_missing_N.txt,
        or 'commands.txt' if none exist (and required=False)
        
    Raises:
        FileNotFoundError: If required=True and no commands_missing_N.txt found
    """
    existing = glob.glob(os.path.join(working_dir, 'commands_missing_*.txt'))
    if not existing:
        if required:
            raise FileNotFoundError(f"No commands_missing_N.txt found in {working_dir}")
        return 'commands.txt'
    nums = []
    for e in existing:
        base = os.path.basename(e)
        try:
            num = int(base.replace('commands_missing_', '').replace('.txt', ''))
            nums.append(num)
        except Exception:
            continue
    if nums:
        return f'commands_missing_{max(nums)}.txt'
    if required:
        raise FileNotFoundError(f"No valid commands_missing_N.txt found in {working_dir}")
    return 'commands.txt'
