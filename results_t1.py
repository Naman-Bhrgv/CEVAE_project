import os
import subprocess

# Models and datasets to iterate through
models = ["cevae", "cvae", "hvae", "vqvae"]
datasets = ["ihdp", "synthetic", "jobs", "twins"]
output_file = "results_scores.txt"

# Parameters
epochs = 100
batch_size = 128
learning_rate = 1e-5
replications = 5

# Prepare results file
with open(output_file, "w") as f:
    f.write("Final Results for all model-dataset combinations:\n\n")

# Iterate through models and datasets
for model in models:
    for dataset in datasets:
        print(f"Running model: {model}, dataset: {dataset}")

        # Command to run the main script
        command = [
            "python", "main.py",
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate),
            "--model", model,
            "--dataset", dataset,
            "--replications", str(replications)
        ]

        try:
            # Run the command and capture output
            result = subprocess.run(command, capture_output=True, text=True)
            output = result.stdout

            # Extract the final results from the output
            final_result = ""
            for line in output.splitlines():
                if (
                    line.strip().startswith("#TRAIN#")
                    or line.strip().startswith("#TEST#")
                    or "BEST" in line
                    or "Policy Risk" in line
                ):
                    final_result += line.strip() + "\n"

            # Save the final results to the file
            with open(output_file, "a") as f:
                f.write(f"Model: {model}, Dataset: {dataset}\n")
                f.write(final_result)
                f.write("\n" + "="*80 + "\n")

        except Exception as e:
            # Log any exception during execution
            with open(output_file, "a") as f:
                f.write(f"Failed for Model: {model}, Dataset: {dataset}\n")
                f.write(f"Error: {str(e)}\n")
                f.write("\n" + "="*80 + "\n")