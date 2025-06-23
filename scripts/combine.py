import os

output_txt = "all_evals_summary.txt"
results_dir = "energy"

with open(output_txt, "w", encoding="utf-8") as out_file:
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == "eval_results.csv":
                full_path = os.path.join(root, file)
                out_file.write(f"==== Path: {full_path} ====\n")
                with open(full_path, "r", encoding="utf-8") as csv_file:
                    out_file.write(csv_file.read())
                out_file.write("\n\n")
print(f"Written all train_evals.csv contents into {output_txt}")
