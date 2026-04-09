import os
import re
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


def _canonical_row_token(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", text).lower()


def _build_row_aliases(row_names: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for name in row_names:
        aliases[_canonical_row_token(name)] = name

        parts = [p for p in re.split(r"\s+", name.strip()) if p]
        if len(parts) > 1:
            aliases[_canonical_row_token(" ".join(reversed(parts)))] = name
    return aliases


def _parse_row_order_spec(row_order: str, row_aliases: dict[str, str]) -> list[str | None]:
    tokens = row_order.split()
    if len(tokens) == 0:
        raise ValueError("row_order must not be empty")

    out: list[str | None] = []
    seen: set[str] = set()
    for tok in tokens:
        if tok == "|":
            out.append(None)
            continue
        if tok == "||":
            out.append("DOUBLE_HLINE")
            continue

        canonical = _canonical_row_token(tok)
        if canonical not in row_aliases:
            raise ValueError(
                f"row_order token '{tok}' not recognized. "
                f"Known rows: {sorted(set(row_aliases.values()))}"
            )
        resolved = row_aliases[canonical]
        if resolved in seen:
            raise ValueError(f"row_order references row '{resolved}' more than once")
        seen.add(resolved)
        out.append(resolved)
    return out


def _normalize_bin_splits(raw_splits: Any, bin_labels: list[str]) -> list[dict[str, Any]]:
    if not isinstance(raw_splits, list) or len(raw_splits) == 0:
        raise ValueError("'bin_splits' must be a non-empty list")

    label_to_index: dict[str, int] = {}
    for i, lbl in enumerate(bin_labels):
        if lbl not in label_to_index:
            label_to_index[lbl] = i

    normalized: list[dict[str, Any]] = []
    for i, split in enumerate(raw_splits):
        if not isinstance(split, dict):
            raise ValueError(f"bin_splits[{i}] must be an object")
        _require_keys(split, ["bins"], context=f"bin_splits[{i}]")
        if "label" in split and not isinstance(split["label"], str):
            raise ValueError(f"bin_splits[{i}].label must be a string")
        if "bins_group_label" in split and not isinstance(split["bins_group_label"], str):
            raise ValueError(f"bin_splits[{i}].bins_group_label must be a string")

        bins = split["bins"]
        if not isinstance(bins, list) or len(bins) == 0:
            raise ValueError(f"bin_splits[{i}].bins must be a non-empty list")

        indices: list[int] = []
        for j, b in enumerate(bins):
            if isinstance(b, int):
                if b < 0 or b >= len(bin_labels):
                    raise ValueError(
                        f"bin_splits[{i}].bins[{j}]={b} out of range [0, {len(bin_labels) - 1}]"
                    )
                indices.append(b)
            elif isinstance(b, str):
                if b not in label_to_index:
                    raise ValueError(
                        f"bin_splits[{i}].bins[{j}] label '{b}' not found in bin_labels"
                    )
                indices.append(label_to_index[b])
            else:
                raise ValueError(
                    f"bin_splits[{i}].bins[{j}] must be an integer index or bin label string"
                )

        normalized.append(
            {
                "label": split.get("label", f"part_{i + 1}"),
                "indices": indices,
                "bins_group_label": split.get("bins_group_label", None),
            }
        )

    return normalized


def _slice_table_by_bins(
    headers: list[str],
    rows: list[list[str] | None | str],
    bin_indices: list[int],
) -> tuple[list[str], list[list[str] | None | str]]:
    sliced_headers = [headers[0], *[headers[i + 1] for i in bin_indices]]
    sliced_rows: list[list[str] | None | str] = []
    for row in rows:
        if row is None or row == "DOUBLE_HLINE":
            sliced_rows.append(row)
            continue
        sliced_rows.append([row[0], *[row[i + 1] for i in bin_indices]])
    return sliced_headers, sliced_rows


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
    if "latex_bins_label" in meta and not isinstance(meta["latex_bins_label"], str):
        raise ValueError("meta.latex_bins_label must be a string")

    if not isinstance(cfg["datasets"], list) or len(cfg["datasets"]) == 0:
        raise ValueError("'datasets' must be a non-empty list")
    if not isinstance(cfg["base_cuts"], list):
        raise ValueError("'base_cuts' must be a list")
    if not isinstance(cfg["weight_variable"], str):
        raise ValueError("'weight_variable' must be a string")
    if "alternative_weight" in cfg and not isinstance(cfg["alternative_weight"], str):
        raise ValueError("'alternative_weight' must be a string when provided")
    if "row_order" in cfg and not isinstance(cfg["row_order"], str):
        raise ValueError("'row_order' must be a string when provided")

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
    use_alternative_weight_requested = False
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

        use_alternative_weight = dscfg.get("use_alternative_weight", False)
        if not isinstance(use_alternative_weight, bool):
            raise ValueError(
                f"datasets[{i}].use_alternative_weight must be a boolean when provided"
            )
        use_alternative_weight_requested = (
            use_alternative_weight_requested or use_alternative_weight
        )

    if use_alternative_weight_requested and "alternative_weight" not in cfg:
        raise ValueError(
            "'alternative_weight' must be provided when any dataset sets "
            "'use_alternative_weight' to true"
        )

    for i, cut_expr in enumerate(cfg["base_cuts"]):
        if not isinstance(cut_expr, str):
            raise ValueError(f"base_cuts[{i}] must be a string")

    bins = _normalize_bin_config(cfg)
    for i, b in enumerate(bins):
        for j, cut_expr in enumerate(b["cuts"]):
            if not isinstance(cut_expr, str):
                raise ValueError(f"bins[{i}].cuts[{j}] must be a string")

    if "bin_splits" in cfg:
        _normalize_bin_splits(cfg["bin_splits"], [b["label"] for b in bins])

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


def _render_table_with_separators(headers: list[str], rows: list[list[str] | None | str]) -> str:
    concrete_rows = [row for row in rows if isinstance(row, list)]
    widths = [len(h) for h in headers]
    for row in concrete_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    header_line = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    rule_line = "-" * len(header_line)
    double_rule_line = "=" * len(header_line)

    lines = [header_line, sep_line]
    for row in rows:
        if row is None:
            lines.append(rule_line)
            continue
        if row == "DOUBLE_HLINE":
            lines.append(double_rule_line)
            continue
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def _escape_latex_plain(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def _escape_latex(text: str) -> str:
    # Preserve content inside $...$ or $$...$$ blocks and only escape plain-text spans.
    out_parts: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        if text[i] == "$":
            delim = "$$" if (i + 1 < n and text[i + 1] == "$") else "$"
            start = i
            i += len(delim)
            end = text.find(delim, i)
            if end == -1:
                # Unmatched delimiter; treat the remainder as plain text.
                out_parts.append(_escape_latex_plain(text[start:]))
                break

            out_parts.append(text[start : end + len(delim)])
            i = end + len(delim)
            continue

        next_dollar = text.find("$", i)
        if next_dollar == -1:
            out_parts.append(_escape_latex_plain(text[i:]))
            break

        out_parts.append(_escape_latex_plain(text[i:next_dollar]))
        i = next_dollar

    return "".join(out_parts)


def _format_latex_label(text: str) -> str:
    # Row/column labels that contain math/comparison tokens are rendered in math mode.
    if any(sym in text for sym in ["<", ">", "="]) or re.search(r"\binf\b", text, flags=re.IGNORECASE):
        math_text = text
        math_text = math_text.replace("<=", r"\le ").replace(">=", r"\ge ")
        math_text = re.sub(r"\binf\b", r"\\infty", math_text, flags=re.IGNORECASE)
        return f"${math_text}$"
    return _escape_latex(text)


def _render_tabular_latex(
    headers: list[str],
    rows: list[list[str] | None | str],
    bins_group_label: str | None = None,
) -> str:
    if len(headers) < 2:
        raise ValueError("LaTeX table requires at least one bin column")

    bin_colspec = "|".join(["r"] * (len(headers) - 1))
    colspec = f"l||{bin_colspec}|"
    lines = [
        rf"\begin{{tabular}}{{{colspec}}}",
    ]

    if bins_group_label is not None:
        lines.extend(
            [
                ""
                + " & "
                + rf"\multicolumn{{{len(headers) - 1}}}{{c|}}{{{_escape_latex(bins_group_label)}}}"
                + r" \\",
                rf"\cline{{2-{len(headers)}}}",
            ]
        )

    lines.extend(
        [
            " & ".join(_format_latex_label(h) for h in headers) + r" \\",
            r"\hline",
        ]
    )

    for row in rows:
        if row is None:
            lines.append(r"\hline")
            continue
        if row == "DOUBLE_HLINE":
            lines.append(r"\hline\hline")
            continue
        formatted_cells = []
        for idx, cell in enumerate(row):
            if idx == 0:
                formatted_cells.append(_format_latex_label(cell))
            else:
                formatted_cells.append(_escape_latex(cell))
        lines.append(" & ".join(formatted_cells) + r" \\")

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
        ]
    )
    return "\n".join(lines)


def _render_table_latex(
    headers: list[str],
    rows: list[list[str] | None | str],
    caption: str | None = None,
    label_source: str | None = None,
    bins_group_label: str | None = None,
) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        _render_tabular_latex(headers, rows, bins_group_label=bins_group_label),
    ]

    if caption:
        lines.append(rf"\caption{{{_escape_latex(caption)}}}")

    label_key = label_source if label_source else "yield_table"
    label_key = re.sub(r"[^a-zA-Z0-9]+", "_", label_key).strip("_").lower()
    if not label_key:
        label_key = "yield_table"
    lines.append(rf"\label{{tab:{label_key}}}")
    lines.append(r"\end{table}")
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
                "use_alternative_weight": dscfg.get("use_alternative_weight", False),
                "key": dscfg["key"],
                "label": dscfg.get("label", dscfg["key"]),
            }
        )
    return built


def run_yield_table(cfg: dict[str, Any], output_format: str = "text") -> str:
    _validate_config(cfg)

    if output_format not in ["text", "latex"]:
        raise ValueError("output_format must be either 'text' or 'latex'")

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
    default_weight = parse_var(cfg["weight_variable"])
    alternative_weight = None
    if "alternative_weight" in cfg:
        alternative_weight = parse_var(cfg["alternative_weight"])

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
            dataset_weight = default_weight
            if info["use_alternative_weight"]:
                dataset_weight = alternative_weight

            if table_metric == "efficiency":
                numerator_cut = splt.cut.AndCuts([efficiency_numerator_cut, bin_cut, info["extra_cut"]])
                denominator_cut = splt.cut.AndCuts([base_cut, bin_cut, info["extra_cut"]])
            else:
                numerator_cut = splt.cut.AndCuts([base_cut, bin_cut, info["extra_cut"]])
                denominator_cut = splt.cut.NoCut()

            num = info["dataset"].estimate_yield(numerator_cut, dataset_weight)
            den = info["dataset"].estimate_yield(denominator_cut, dataset_weight) if table_metric == "efficiency" else 0.0
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
    dataset_row_map: dict[str, list[str]] = {}
    total_row_map: dict[str, list[str]] = {}

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
        dataset_row_map[info["key"]] = row

    if totals:
        for total_cfg in totals:
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
            total_row_map[total_cfg["label"]] = total_row

    table_rows: list[list[str] | None | str] = []
    row_aliases = _build_row_aliases([*dataset_row_map.keys(), *total_row_map.keys()])

    if "row_order" in cfg:
        row_sequence = _parse_row_order_spec(cfg["row_order"], row_aliases)
        for item in row_sequence:
            if item is None:
                table_rows.append(None)
            elif item == "DOUBLE_HLINE":
                table_rows.append("DOUBLE_HLINE")
            elif item in dataset_row_map:
                table_rows.append(dataset_row_map[item])
            elif item in total_row_map:
                table_rows.append(total_row_map[item])
            else:
                raise ValueError(f"Internal error: unresolved row '{item}'")
    else:
        for row in dataset_row_map.values():
            table_rows.append(row)
        if len(total_row_map) > 0:
            table_rows.append(None)
            total_names = list(total_row_map.keys())
            for idx, tname in enumerate(total_names):
                table_rows.append(total_row_map[tname])
                if idx < len(total_names) - 1:
                    table_rows.append(None)

    split_specs = None
    if "bin_splits" in cfg:
        split_specs = _normalize_bin_splits(cfg["bin_splits"], row_labels)

    if output_format == "latex":
        label_source = str(meta["table_name"]) if "table_name" in meta else os.path.splitext(os.path.basename(output_file))[0]
        if split_specs is None:
            output_text = _render_table_latex(
                headers,
                table_rows,
                caption=str(meta["table_name"]) if "table_name" in meta else None,
                label_source=label_source,
                bins_group_label=meta.get("latex_bins_label"),
            ) + "\n"
        else:
            tabular_blocks: list[str] = []
            for i, split in enumerate(split_specs):
                split_headers, split_rows = _slice_table_by_bins(headers, table_rows, split["indices"])
                split_bins_label = (
                    split["bins_group_label"]
                    if split["bins_group_label"] is not None
                    else meta.get("latex_bins_label")
                )
                block_lines: list[str] = [rf"\textbf{{{_escape_latex(str(split['label']))}}}\\"]
                block_lines.append(
                    _render_tabular_latex(
                        split_headers,
                        split_rows,
                        bins_group_label=split_bins_label,
                    )
                )
                tabular_blocks.append("\n".join(block_lines))

            block_sep = "\n\n\\vspace{0.8em}\n\n"
            lines = [r"\begin{table}[htbp]", r"\centering", block_sep.join(tabular_blocks)]
            if "table_name" in meta:
                lines.append(rf"\caption{{{_escape_latex(str(meta['table_name']))}}}")
            lines.append(rf"\label{{tab:{re.sub(r'[^a-zA-Z0-9]+', '_', label_source).strip('_').lower() or 'yield_table'}}}")
            lines.append(r"\end{table}")
            output_text = "\n".join(lines).rstrip() + "\n"
    else:
        if split_specs is None:
            output_blocks = []
            if "table_name" in meta:
                output_blocks.append(str(meta["table_name"]))
                output_blocks.append("")

            output_blocks.append(_render_table_with_separators(headers, table_rows))
            output_text = "\n".join(output_blocks).rstrip() + "\n"
        else:
            output_blocks = []
            for split in split_specs:
                split_headers, split_rows = _slice_table_by_bins(headers, table_rows, split["indices"])
                if "table_name" in meta:
                    output_blocks.append(f"{meta['table_name']} ({split['label']})")
                else:
                    output_blocks.append(str(split["label"]))
                output_blocks.append("")
                output_blocks.append(_render_table_with_separators(split_headers, split_rows))
                output_blocks.append("")
            output_text = "\n".join(output_blocks).rstrip() + "\n"

    outdir = os.path.dirname(output_file)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(output_text)

    return output_file
