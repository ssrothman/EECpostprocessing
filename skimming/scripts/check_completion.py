#!/usr/bin/env python

import argparse
import os
import sys

from general.datasets.datasets import get_target_files
from general.fslookup.skim_path import lookup_skim_path


def resolve_objsyst(dataset: str, objsyst: str | None) -> str:
	if objsyst is not None:
		return objsyst
	if "data" in dataset.lower():
		return "DATA"
	return "nominal"


def check_one_table(
	runtag: str,
	dataset: str,
	objsyst: str,
	location: str,
	configsuite: str,
	table_name: str,
	n_targets: int,
) -> dict:
	expected_targets = 2 * n_targets if table_name == "count" else n_targets

	fs, path = lookup_skim_path(
		location=location,
		configsuite=configsuite,
		runtag=runtag,
		dataset=dataset,
		objsyst=objsyst,
		table=table_name,
	)

	parquet_files = fs.glob(os.path.join(path, "*.parquet"))
	json_files = fs.glob(os.path.join(path, "*.json"))
	json_files = [f for f in json_files if not f.endswith("merged.json")]

	n_parquet = len(parquet_files)
	n_json = len(json_files)
	best_count = max(n_parquet, n_json)
	complete = (n_parquet == expected_targets) or (n_json == expected_targets)

	return {
		"table_name": table_name,
		"path": path,
		"expected_targets": expected_targets,
		"n_parquet": n_parquet,
		"n_json": n_json,
		"best_count": best_count,
		"complete": complete,
	}


def print_single_result(
	runtag: str,
	dataset: str,
	objsyst: str,
	location: str,
	configsuite: str,
	n_targets: int,
	result: dict,
) -> None:
	print("Completion check")
	print(f"  runtag           : {runtag}")
	print(f"  dataset          : {dataset}")
	print(f"  objsyst          : {objsyst}")
	print(f"  table (resolved) : {result['table_name']}")
	print(f"  location         : {location}")
	print(f"  configsuite      : {configsuite}")
	print(f"  output path      : {result['path']}")
	print(f"  targets (base)   : {n_targets}")
	print(f"  targets (expect) : {result['expected_targets']}")
	print(f"  parquet outputs  : {result['n_parquet']}")
	print(f"  json outputs     : {result['n_json']}")
	print(f"  completed        : {result['complete']}")

	if result["expected_targets"] > 0:
		frac = 100.0 * result["best_count"] / result["expected_targets"]
		print(
			f"  progress         : {result['best_count']}/{result['expected_targets']} ({frac:.1f}%)"
		)


def print_multi_results(
	runtag: str,
	dataset: str,
	objsyst: str,
	location: str,
	configsuite: str,
	n_targets: int,
	results: list[dict],
) -> None:
	print("Completion check (multi-table)")
	print(f"  runtag      : {runtag}")
	print(f"  dataset     : {dataset}")
	print(f"  objsyst     : {objsyst}")
	print(f"  location    : {location}")
	print(f"  configsuite : {configsuite}")
	print(f"  targets     : {n_targets} (base; 'count' expects 2x)")
	print()
	print("status  progress          table")
	print("------  ----------------  ------------------------------")
	for result in results:
		expected = result["expected_targets"]
		if expected > 0:
			frac = 100.0 * result["best_count"] / expected
			prog = f"{result['best_count']}/{expected} ({frac:5.1f}%)"
		else:
			prog = f"{result['best_count']}/{expected}"
		status = "PASS" if result["complete"] else "FAIL"
		print(f"{status:<6}  {prog:<16}  {result['table_name']}")


def main() -> int:
	parser = argparse.ArgumentParser(
		"check output file completion for one or more skim tables"
	)
	parser.add_argument("runtag", type=str, help="Runtag")
	parser.add_argument("dataset", type=str, help="Dataset name")
	parser.add_argument(
		"table",
		type=str,
		nargs="?",
		default=None,
		help="Single table name or table class string",
	)
	parser.add_argument(
		"--tables",
		type=str,
		nargs="+",
		help="Check multiple tables in one call",
	)
	parser.add_argument(
		"--objsyst",
		type=str,
		default=None,
		help="Object systematic variation (default: DATA for data datasets, nominal otherwise)",
	)
	parser.add_argument(
		"--location",
		type=str,
		default="local-submit",
		help="Storage location key",
	)
	parser.add_argument(
		"--configsuite",
		type=str,
		default="BasicConfig",
		help="Configuration suite name",
	)
	parser.add_argument(
		"--include-dropped",
		action="store_true",
		help="Include dropped input files in the target file count",
	)
	parser.add_argument(
		"--strict",
		action="store_true",
		help="Exit with code 1 when incomplete (useful for automation)",
	)

	args = parser.parse_args()

	if args.table is None and not args.tables:
		parser.error("provide either positional 'table' or '--tables ...'")

	if args.table is not None and args.tables:
		parser.error("use either positional 'table' or '--tables ...', not both")

	objsyst = resolve_objsyst(args.dataset, args.objsyst)

	exclude_dropped = not args.include_dropped
	target_files, _ = get_target_files(
		args.runtag,
		args.dataset,
		exclude_dropped=exclude_dropped,
	)
	n_targets = len(target_files)

	if args.tables:
		results = [
			check_one_table(
				runtag=args.runtag,
				dataset=args.dataset,
				objsyst=objsyst,
				location=args.location,
				configsuite=args.configsuite,
				table_name=table,
				n_targets=n_targets,
			)
			for table in args.tables
		]
		print_multi_results(
			runtag=args.runtag,
			dataset=args.dataset,
			objsyst=objsyst,
			location=args.location,
			configsuite=args.configsuite,
			n_targets=n_targets,
			results=results,
		)
		any_incomplete = any(not result["complete"] for result in results)
		if args.strict and any_incomplete:
			return 1
		return 0

	result = check_one_table(
		runtag=args.runtag,
		dataset=args.dataset,
		objsyst=objsyst,
		location=args.location,
		configsuite=args.configsuite,
		table_name=args.table,
		n_targets=n_targets,
	)
	print_single_result(
		runtag=args.runtag,
		dataset=args.dataset,
		objsyst=objsyst,
		location=args.location,
		configsuite=args.configsuite,
		n_targets=n_targets,
		result=result,
	)
	if args.strict and not result["complete"]:
		return 1
	return 0


if __name__ == "__main__":
	sys.exit(main())
