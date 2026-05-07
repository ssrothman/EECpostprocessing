#!/usr/bin/env python

from __future__ import annotations

import argparse
import filecmp
import sys
import shutil
from pathlib import Path

HISTOGRAM_SIGNATURE_FILES = {"values.npy", "cov.npy", "bincfg.json"}
DETECTORMODEL_SIGNATURE_FILES = {
	"transfer0.npy",
	"gamma0.npy",
	"rho0.npy",
	"transferVariations.npy",
	"transferVarIndices.npy",
	"gammaVariations.npy",
	"rhoVariations.npy",
	"binning.json",
	"nuisance_names.txt",
}


def _has_signature(path: Path, signature_files: set[str]) -> bool:
	return path.is_dir() and signature_files.issubset({child.name for child in path.iterdir()})


def _classify_workspace_object(path: Path) -> str | None:
	is_histogram = _has_signature(path, HISTOGRAM_SIGNATURE_FILES)
	is_model = _has_signature(path, DETECTORMODEL_SIGNATURE_FILES)

	if is_histogram and not is_model:
		return "histogram"
	if is_model and not is_histogram:
		return "detectormodel"
	return None


def _sync_workspace_config(source_workspace: Path, target_workspace: Path) -> None:
	source_config = source_workspace / "config.json"
	if not source_config.is_file():
		raise FileNotFoundError(f"Source workspace is missing config.json: {source_config}")

	target_config = target_workspace / "config.json"
	if target_config.is_file() and not filecmp.cmp(source_config, target_config, shallow=False):
		raise ValueError(
            f"Target workspace already has a config.json that differs from the source: {target_config}"
        )
	elif target_config.is_file():
		print(f"Target workspace already has an identical config.json; skipping copy: {target_config}")
		return

	shutil.copyfile(source_config, target_config)


def _copy_rebinned_histogram(source: Path, destination: Path, rebinning_spec: Path) -> None:
	from unfolding.histogram import Histogram

	histogram = Histogram.from_disk(str(source))
	histogram.rebin(str(rebinning_spec))
	histogram.dump_to_disk(str(destination))


def _copy_rebinned_model(source: Path, destination: Path, rebinning_reco: Path, rebinning_gen: Path) -> None:
	from unfolding.detectormodel import DetectorModel

	model = DetectorModel.from_disk(str(source))
	model.rebin(str(rebinning_reco), str(rebinning_gen))
	model.dump_to_disk(str(destination))


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Rebin selected objects from an existing unfolding workspace into a new workspace."
	)
	parser.add_argument(
		"workspace",
		help="Path to the existing unfolding workspace",
	)
	parser.add_argument(
		"--rebinning-gen",
		default="rebinning_gen.json",
		help="Path to the gen-level rebinning spec JSON file (default: rebinning_gen.json)",
	)
	parser.add_argument(
		"--rebinning-reco",
		default="rebinning_reco.json",
		help="Path to the reco-level rebinning spec JSON file (default: rebinning_reco.json)",
	)
	parser.add_argument(
		"--output",
		default="./",
		help="Path to write the new rebinned workspace (default: ./)",
	)
	parser.add_argument(
		"--objects",
		nargs="+",
		default=["gen", "reco", "model"],
		help="Objects to rebin from the workspace (default: gen reco model)",
	)
	args = parser.parse_args()

	workspace = Path(args.workspace).expanduser().resolve()
	rebinning_gen = Path(args.rebinning_gen).expanduser().resolve()
	rebinning_reco = Path(args.rebinning_reco).expanduser().resolve()
	output_workspace = Path(args.output).expanduser().resolve()

	if not workspace.is_dir():
		raise NotADirectoryError(f"Input workspace is not a directory: {workspace}")
	if not rebinning_gen.is_file():
		raise FileNotFoundError(f"Gen rebinning spec does not exist: {rebinning_gen}")
	if not rebinning_reco.is_file():
		raise FileNotFoundError(f"Reco rebinning spec does not exist: {rebinning_reco}")

	output_workspace.mkdir(parents=True, exist_ok=True)
	_sync_workspace_config(workspace, output_workspace)

	for object_name in args.objects:
		source_dir = workspace / object_name
		destination_dir = output_workspace / object_name

		if not source_dir.is_dir():
			print(f"Warning: workspace does not contain an '{object_name}' directory at {source_dir}; skipping")
			continue

		object_kind = _classify_workspace_object(source_dir)
		if object_kind is None:
			print(
				f"Warning: could not identify '{object_name}' at {source_dir} as a Histogram or DetectorModel; skipping"
			)
			continue

		if object_kind == "histogram":
			if 'gen' not in str(object_name).lower() and 'reco' not in str(object_name).lower():
				print(
					f"Warning: '{object_name}' looks like a Histogram but does not contain 'gen' or 'reco' in its name to indicate which rebinning spec to use; skipping"
				)
				continue

			rebinning_spec = rebinning_gen if 'gen' in str(object_name).lower() else rebinning_reco
			_copy_rebinned_histogram(source_dir, destination_dir, rebinning_spec)
			print(f"Rebinned histogram '{object_name}' -> {destination_dir}")
			continue

		if object_name != "model":
			print(
				f"Warning: '{object_name}' looks like a DetectorModel but is not the standard model workspace object; skipping"
			)
			continue

		_copy_rebinned_model(source_dir, destination_dir, rebinning_reco, rebinning_gen)
		print(f"Rebinned detector model '{object_name}' -> {destination_dir}")


if __name__ == "__main__":
	main()