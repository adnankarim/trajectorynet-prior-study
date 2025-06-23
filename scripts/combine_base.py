import os

csv_paths = [
    "results/tree_baselines.csv",
    "results/circle5_baselines.csv",
    "results/cycle_baselines.csv"
]

output_txt = "all_baseline_evals.txt"

with open(output_txt, "w", encoding="utf-8") as out_file:
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"[WARNING] Missing: {path}")
            continue

        out_file.write(f"==== Path: {os.path.abspath(path)} ====\n")

        with open(path, "r", encoding="utf-8") as csv_file:
            out_file.write(csv_file.read())

        out_file.write("\n\n")

print(f"✓ Combined output written to: {output_txt}")
