import argparse
import json
import os

from plotting.yielddriver import run_yield_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JSON-driven yield table scripts.")
    parser.add_argument(
        "configs",
        type=str,
        nargs="+",
        help="Path(s) to yield table configuration file(s) (JSON format).",
    )
    parser.add_argument(
        "--tex",
        action="store_true",
        help="Write LaTeX-formatted tables (.tex output).",
    )
    args = parser.parse_args()

    for config_path in args.configs:
        with open(config_path, "r") as f:
            cfg = json.load(f)

        output_format = "latex" if args.tex else "text"
        if args.tex:
            outpath = cfg["meta"].get("output_file", "")
            if isinstance(outpath, str) and outpath != "":
                root, _ = os.path.splitext(outpath)
                cfg["meta"]["output_file"] = root + ".tex"

        outfile = run_yield_table(cfg, output_format=output_format)
        print(f"[{config_path}] Wrote yield table to: {outfile}")


if __name__ == "__main__":
    main()
