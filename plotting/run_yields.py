import argparse
import json

from plotting.yielddriver import run_yield_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JSON-driven yield table scripts.")
    parser.add_argument(
        "configs",
        type=str,
        nargs="+",
        help="Path(s) to yield table configuration file(s) (JSON format).",
    )
    args = parser.parse_args()

    for config_path in args.configs:
        with open(config_path, "r") as f:
            cfg = json.load(f)

        outfile = run_yield_table(cfg)
        print(f"[{config_path}] Wrote yield table to: {outfile}")


if __name__ == "__main__":
    main()
