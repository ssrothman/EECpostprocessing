#!/usr/bin/env python3

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

def parquet_row_count(path: Path) -> int:
	# detect that the file is empty if it is exactly 178 bytes, which is the size of an empty parquet file 
	if path.stat().st_size == 178:
		return 0
	else:
		return 100
	#pf = pq.ParquetFile(path)
	#return pf.metadata.num_rows


def main() -> int:
	parser = argparse.ArgumentParser(description="Delete empty parquet files in one or more folders")
	parser.add_argument("folders", nargs="+", type=str, help="Folders containing parquet files")
	args = parser.parse_args()

	total_checked = 0
	total_deleted = 0
	total_kept = 0

	for folder_arg in args.folders:
		folder = Path(folder_arg).resolve()
		if not folder.is_dir():
			raise RuntimeError(f"Folder does not exist or is not a directory: {folder}")

		parquet_files = sorted(folder.glob("*.parquet"))
		checked = 0
		deleted = 0
		kept = 0

		for path in tqdm(parquet_files, desc=f"Checking parquet files in {folder.name}"):
			checked += 1
			row_count = parquet_row_count(path)
			if row_count == 0:
				path.unlink()
				deleted += 1
				#print(f"deleted empty parquet: {path}")
			else:
				kept += 1

		print(f"folder: {folder}")
		print(f"  checked {checked} parquet files")
		print(f"  deleted {deleted} empty parquet files")
		print(f"  kept {kept} non-empty parquet files")

		total_checked += checked
		total_deleted += deleted
		total_kept += kept

	print(f"total checked {total_checked} parquet files")
	print(f"total deleted {total_deleted} empty parquet files")
	print(f"total kept {total_kept} non-empty parquet files")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())