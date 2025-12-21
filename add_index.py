import argparse
import os

import pandas as pd


def add_index_column(
	input_path: str,
	output_path: str,
	sep: str = ";",
	index_name: str = "index",
	start: int = 0,
) -> None:
	df = pd.read_csv(input_path, sep=sep)

	if index_name in df.columns:
		raise ValueError(f"Column {index_name!r} already exists in {input_path}")

	df.insert(0, index_name, range(start, start + len(df)))
	df.to_csv(output_path, sep=sep, index=False)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Add an index column as the first column to initial_results.csv."
	)
	parser.add_argument(
		"--input",
		default="initial_results.csv",
		help="Input CSV path (default: initial_results.csv)",
	)
	parser.add_argument(
		"--output",
		default=None,
		help="Output CSV path (default: overwrite input)",
	)
	parser.add_argument(
		"--sep",
		default=";",
		help="CSV separator (default: ';')",
	)
	parser.add_argument(
		"--name",
		default="index",
		help="Name of the new index column (default: index)",
	)
	parser.add_argument(
		"--start",
		type=int,
		default=0,
		help="Starting index value (default: 0)",
	)
	args = parser.parse_args()

	input_path = args.input
	output_path = args.output or args.input

	if not os.path.exists(input_path):
		raise FileNotFoundError(f"Input file not found: {input_path}")

	add_index_column(
		input_path=input_path,
		output_path=output_path,
		sep=args.sep,
		index_name=args.name,
		start=args.start,
	)
	print(f"Wrote {output_path} (added column {args.name!r} as first column)")


if __name__ == "__main__":
	main()

