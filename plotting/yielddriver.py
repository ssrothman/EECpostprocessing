import os
from typing import Any

import simonplot as splt

from plotting.plotdriver import build_dataset_from_dscfg, parse_var


def _require_keys(obj: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ValueError(f"Missing required key(s) {missing} in {context}")


def _parse_cut_strings(cut_strings: list[str], context: str):
    if len(cut_strings) == 0:
        return splt.cut.NoCut()

    parsed = []
    for idx, cut_expr in enumerate(cut_strings):
        if not isinstance(cut_expr, str):
            raise ValueError(f"{context}[{idx}] must be a string cut expression")
        parsed_cut = parse_var(cut_expr)
        parsed.append(parsed_cut)
    return splt.cut.AndCuts(parsed)


def _normalize_bin_config(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    bins = cfg.get("bins")
    if not isinstance(bins, list) or len(bins) == 0:
        raise ValueError("'bins' must be a non-empty list")

    if all(isinstance(b, list) for b in bins):
        labels = cfg.get("bin_labels", [])
        if labels and len(labels) != len(bins):
            raise ValueError("'bin_labels' must have the same length as 'bins'")

        normalized: list[dict[str, Any]] = []
        for i, cut_list in enumerate(bins):
            if not isinstance(cut_list, list):
                raise ValueError(f"bins[{i}] must be a list of cut strings")
            normalized.append(
                {
                    "label": labels[i] if labels else f"bin_{i}",
                    "cuts": cut_list,
                }
            )
        return normalized

    normalized = []
    for i, b in enumerate(bins):
        if not isinstance(b, dict):
            raise ValueError(
                "'bins' entries must be either list-of-cuts or objects with 'label' and 'cuts'"
            )
        _require_keys(b, ["label", "cuts"], context=f"bins[{i}]")
        if not isinstance(b["label"], str):
            raise ValueError(f"bins[{i}].label must be a string")
        if not isinstance(b["cuts"], list):
            raise ValueError(f"bins[{i}].cuts must be a list")
        normalized.append(b)
    return normalized


def _normalize_shared_cutlist(raw_cuts: Any, context: str) -> list[str]:
    if not isinstance(raw_cuts, list):
        raise ValueError(f"'{context}' must be a list of cut strings")

    normalized: list[str] = []
    for i, cut_expr in enumerate(raw_cuts):
        if not isinstance(cut_expr, str):
            raise ValueError(f"{context}[{i}] must be a string")
        normalized.append(cut_expr)
    return normalized


def _validate_config(cfg: dict[str, Any]) -> None:
    _require_keys(
        cfg,
        ["meta", "datasets", "base_cuts", "bins", "weight_variable"],
        context="top-level config",
    )

    meta = cfg["meta"]
    if not isinstance(meta, dict):
        raise ValueError("'meta' must be an object")
    _require_keys(meta, ["output_file", "target_lumi"], context="meta")

    if not isinstance(meta["output_file"], str):
        raise ValueError("meta.output_file must be a string")
    if not isinstance(meta["target_lumi"], (int, float)):
        raise ValueError("meta.target_lumi must be numeric")

    if not isinstance(cfg["datasets"], list) or len(cfg["datasets"]) == 0:
        raise ValueError("'datasets' must be a non-empty list")
    if not isinstance(cfg["base_cuts"], list):
        raise ValueError("'base_cuts' must be a list")
    if not isinstance(cfg["weight_variable"], str):
        raise ValueError("'weight_variable' must be a string")

    table_metric = cfg.get("table_metric", "yield")
    if table_metric not in ["yield", "efficiency"]:
        raise ValueError("'table_metric' must be either 'yield' or 'efficiency'")

    required_dataset_keys = [
        "key",
        "configsuite",
        "runtag",
        "name",
        "table",
        "objsyst",
        "isstack",
    ]
    dataset_keys: list[str] = []
    for i, dscfg in enumerate(cfg["datasets"]):
        if not isinstance(dscfg, dict):
            raise ValueError(f"datasets[{i}] must be an object")
        _require_keys(dscfg, required_dataset_keys, context=f"datasets[{i}]")
        if not isinstance(dscfg["key"], str):
            raise ValueError(f"datasets[{i}].key must be a string")
        if dscfg["key"] in dataset_keys:
            raise ValueError(f"Duplicate dataset key: {dscfg['key']}")
        dataset_keys.append(dscfg["key"])

        extra_cuts = dscfg.get("extra_cuts", [])
        if not isinstance(extra_cuts, list):
            raise ValueError(f"datasets[{i}].extra_cuts must be a list")
        for j, cut_expr in enumerate(extra_cuts):
            if not isinstance(cut_expr, str):
                raise ValueError(f"datasets[{i}].extra_cuts[{j}] must be a string")

    for i, cut_expr in enumerate(cfg["base_cuts"]):
        if not isinstance(cut_expr, str):
            raise ValueError(f"base_cuts[{i}] must be a string")

    bins = _normalize_bin_config(cfg)
    for i, b in enumerate(bins):
        for j, cut_expr in enumerate(b["cuts"]):
            if not isinstance(cut_expr, str):
                raise ValueError(f"bins[{i}].cuts[{j}] must be a string")

    if "alternative_cut" in cfg:
        _normalize_shared_cutlist(
            cfg["alternative_cut"],
            context="alternative_cut",
        )

    totals = cfg.get("totals", [])
    if not isinstance(totals, list):
        raise ValueError("'totals' must be a list when provided")

    dset_key_set = set(dataset_keys)
    for i, total_cfg in enumerate(totals):
        if not isinstance(total_cfg, dict):
            raise ValueError(f"totals[{i}] must be an object")
        _require_keys(total_cfg, ["label", "datasets"], context=f"totals[{i}]")
        if not isinstance(total_cfg["label"], str):
            raise ValueError(f"totals[{i}].label must be a string")
        if not isinstance(total_cfg["datasets"], list) or len(total_cfg["datasets"]) == 0:
            raise ValueError(f"totals[{i}].datasets must be a non-empty list")
        for ref in total_cfg["datasets"]:
            if ref not in dset_key_set:
                raise ValueError(
                    f"totals[{i}] references unknown dataset key '{ref}'. "
                    f"Known keys: {sorted(dset_key_set)}"
                )

    pct_cfg = cfg.get("percent_contribution")
    if pct_cfg is not None:
        if not isinstance(pct_cfg, dict):
            raise ValueError("'percent_contribution' must be an object when provided")
        _require_keys(pct_cfg, ["reference_total"], context="percent_contribution")
        if not isinstance(pct_cfg["reference_total"], str):
            raise ValueError("percent_contribution.reference_total must be a string")

        total_labels = {t["label"] for t in totals}
        if pct_cfg["reference_total"] not in total_labels:
            raise ValueError(
                "percent_contribution.reference_total must match one totals label. "
                f"Known labels: {sorted(total_labels)}"
            )

        if "format" in pct_cfg and not isinstance(pct_cfg["format"], str):
            raise ValueError("percent_contribution.format must be a string")
        if "label_suffix" in pct_cfg and not isinstance(pct_cfg["label_suffix"], str):
            raise ValueError("percent_contribution.label_suffix must be a string")


def _format_float(value: float, fmt: str) -> str:
    try:
        return fmt % value
    except Exception as exc:
        raise ValueError(f"Invalid float format '{fmt}': {exc}") from exc


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    header_line = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    row_lines = [
        " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in rows
    ]

    return "\n".join([header_line, sep_line, *row_lines])


def _render_table_with_separators(headers: list[str], rows: list[list[str] | None]) -> str:
    concrete_rows = [row for row in rows if row is not None]
    widths = [len(h) for h in headers]
    for row in concrete_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    header_line = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    rule_line = "-" * len(header_line)

    lines = [header_line, sep_line]
    for row in rows:
        if row is None:
            lines.append(rule_line)
            continue
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def _build_datasets(cfg: dict[str, Any]):
    dataset_cfgs = cfg["datasets"]

    first_objsyst = dataset_cfgs[0]["objsyst"]
    all_same_objsyst = all(d["objsyst"] == first_objsyst for d in dataset_cfgs[1:])

    first_extracut = dataset_cfgs[0].get("extra_cuts", [])
    all_same_extracut = all(
        d.get("extra_cuts", []) == first_extracut for d in dataset_cfgs[1:]
    )

    built = []
    for i, dscfg in enumerate(dataset_cfgs):
        dsetcut = _parse_cut_strings(dscfg.get("extra_cuts", []), context=f"datasets[{i}].extra_cuts")
        dataset = build_dataset_from_dscfg(
            dscfg,
            all_same_objsyst=all_same_objsyst,
            all_same_extracut=all_same_extracut,
            dsetcut=dsetcut,
        )
        built.append(
            {
                "cfg": dscfg,
                "dataset": dataset,
                "extra_cut": dsetcut,
                "key": dscfg["key"],
                "label": dscfg.get("label", dscfg["key"]),
            }
        )
    return built


def run_yield_table(cfg: dict[str, Any]) -> str:
    _validate_config(cfg)

    meta = cfg["meta"]
    output_file = meta["output_file"]
    target_lumi = float(meta["target_lumi"])
    float_format = meta.get("float_format", "%.2f")
    table_metric = cfg.get("table_metric", "yield")

    bins = _normalize_bin_config(cfg)
    efficiency_numerator_cut = splt.cut.NoCut()
    if "alternative_cut" in cfg:
        efficiency_numerator_cut = _parse_cut_strings(
            _normalize_shared_cutlist(
                cfg["alternative_cut"],
                context="alternative_cut",
            ),
            context="alternative_cut",
        )

    base_cut = _parse_cut_strings(cfg["base_cuts"], context="base_cuts")
    weight = parse_var(cfg["weight_variable"])

    built_datasets = _build_datasets(cfg)
    for info in built_datasets:
        info["dataset"].compute_weight(target_lumi)

    row_labels = [b["label"] for b in bins]
    numerator_rows: list[list[float]] = []
    denominator_rows: list[list[float]] = []

    for bin_index, b in enumerate(bins):
        bin_cut = _parse_cut_strings(b["cuts"], context=f"bins[{bin_index}].cuts")
        num_values = []
        den_values = []
        for info in built_datasets:
            if table_metric == "efficiency":
                numerator_cut = splt.cut.AndCuts([efficiency_numerator_cut, bin_cut, info["extra_cut"]])
                denominator_cut = splt.cut.AndCuts([base_cut, bin_cut, info["extra_cut"]])
            else:
                numerator_cut = splt.cut.AndCuts([base_cut, bin_cut, info["extra_cut"]])
                denominator_cut = splt.cut.NoCut()

            num = info["dataset"].estimate_yield(numerator_cut, weight)
            den = info["dataset"].estimate_yield(denominator_cut, weight) if table_metric == "efficiency" else 0.0
            num_values.append(float(num))
            den_values.append(float(den))
        numerator_rows.append(num_values)
        denominator_rows.append(den_values)

    if table_metric == "yield":
        table_values = numerator_rows
    else:
        table_values = []
        for bin_idx in range(len(bins)):
            eff_row = []
            for dset_idx in range(len(built_datasets)):
                den = denominator_rows[bin_idx][dset_idx]
                num = numerator_rows[bin_idx][dset_idx]
                eff_row.append(0.0 if den == 0 else num / den)
            table_values.append(eff_row)

    totals = cfg.get("totals", [])
    totals_by_label: dict[str, list[float]] = {}
    key_to_index = {info["key"]: i for i, info in enumerate(built_datasets)}
    for total_cfg in totals:
        if table_metric == "yield":
            totals_by_label[total_cfg["label"]] = [
                sum(
                    table_values[bin_idx][key_to_index[dkey]]
                    for dkey in total_cfg["datasets"]
                )
                for bin_idx in range(len(bins))
            ]
        else:
            totals_by_label[total_cfg["label"]] = [
                0.0
                if sum(
                    denominator_rows[bin_idx][key_to_index[dkey]]
                    for dkey in total_cfg["datasets"]
                )
                == 0
                else sum(
                    numerator_rows[bin_idx][key_to_index[dkey]]
                    for dkey in total_cfg["datasets"]
                )
                /
                sum(
                    denominator_rows[bin_idx][key_to_index[dkey]]
                    for dkey in total_cfg["datasets"]
                )
                for bin_idx in range(len(bins))
            ]

    pct_cfg = cfg.get("percent_contribution")
    pct_format = "%.1f"
    ref_total_values: list[float] | None = None
    if pct_cfg is not None and table_metric == "yield":
        pct_format = pct_cfg.get("format", "%.1f")
        ref_total_label = pct_cfg["reference_total"]
        ref_total_values = totals_by_label[ref_total_label]

    def _format_table_metric(value: float) -> str:
        if table_metric == "efficiency":
            return f"{_format_float(100.0 * value, float_format)}%"
        return _format_float(value, float_format)

    headers = ["Dataset", *row_labels]
    table_rows: list[list[str] | None] = []

    for dset_idx, info in enumerate(built_datasets):
        row = [info["label"]]
        for bin_idx in range(len(bins)):
            yld_val = table_values[bin_idx][dset_idx]
            yld_text = _format_table_metric(yld_val)
            if ref_total_values is not None:
                den = ref_total_values[bin_idx]
                pct = 0.0 if den == 0 else 100.0 * yld_val / den
                pct_text = _format_float(pct, pct_format)
                row.append(f"{yld_text} ({pct_text}%)")
            else:
                row.append(yld_text)
        table_rows.append(row)

    output_blocks = []
    if "table_name" in meta:
        output_blocks.append(str(meta["table_name"]))
        output_blocks.append("")

    if totals:
        table_rows.append(None)
        for idx, total_cfg in enumerate(totals):
            total_values = totals_by_label[total_cfg["label"]]
            total_row = [total_cfg["label"]]
            for bin_idx in range(len(bins)):
                total_text = _format_table_metric(total_values[bin_idx])
                if ref_total_values is not None:
                    den = ref_total_values[bin_idx]
                    pct = 0.0 if den == 0 else 100.0 * total_values[bin_idx] / den
                    pct_text = _format_float(pct, pct_format)
                    total_text = f"{total_text} ({pct_text}%)"
                total_row.append(total_text)
            table_rows.append(total_row)
            if idx < len(totals) - 1:
                table_rows.append(None)

    output_blocks.append(_render_table_with_separators(headers, table_rows))

    output_text = "\n".join(output_blocks).rstrip() + "\n"

    outdir = os.path.dirname(output_file)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(output_text)

    return output_file
