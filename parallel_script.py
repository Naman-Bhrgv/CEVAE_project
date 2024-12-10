import subprocess


# List of scripts with their respective arguments
scripts_with_args = [
    ["main.py", "--epochs", "1", "--batch-size", "128", "--learning-rate","1e-5","--model","vqvae","--dataset","jobs","--replications","2", "--ofile" , "op_jobs_vqvae.txt"],
    ["main.py", "--epochs", "1", "--batch-size", "128", "--learning-rate","1e-5","--model","cevae","--dataset","jobs","--replications","2", "--ofile" , "op_jobs_cevae.txt"]
]

# List of output file names for each script
output_files = ["output_vqvae_jobs.txt","output_vqvae_jobs.txt"]


# Launch each script in parallel with its arguments and redirect output to files
processes = []
for script_with_args, output_file in zip(scripts_with_args, output_files):
    print(output_file)
    with open(output_file, "w") as out_file:
        process = subprocess.Popen(["python"] + script_with_args, stdout=out_file, stderr=subprocess.STDOUT)
        processes.append(process)

# Wait for all scripts to finish
for process in processes:
    process.communicate()

