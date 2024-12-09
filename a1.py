import subprocess

# Define the command
models = ["bvae", "hvae"]
datasets = ["ihdp"]
epochs = [15]
batch_sizes = [64,128]
learning_rates = [1e-5,5e-5,1e-4]
hidden_layers = [128,64,256]
hidden_dims = [100,120,150,200]
wdecay=[1e-5,1e-4,1e-3]

# Iterate through all combinations of parameter values
for model in models:
    for dataset in datasets:
        for epoch in epochs:
            for batch_size in batch_sizes:
                for learning_rate in learning_rates:
                    for hidden_layer in hidden_layers:
                        for hidden_dim in hidden_dims:
                            for wd in wdecay: 
                                # Construct the command dynamically
                                command = (
                                    f"python main.py "
                                    f"--model {model} "
                                    f"--dataset {dataset} "
                                    f"--epochs {epoch} "
                                    f"--batch-size {batch_size} "
                                    f"--learning-rate {learning_rate} "
                                    f"--hidden-layers {hidden_layer} "
                                    f"--hidden-dim {hidden_dim} "
                                    f"--weight-decay {wd}"
                                )
    
                                
                                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True)
    
                                print("Execution started-")
                                # Open the output file for writing
                                with open("output_ht_sv.txt", "a") as output_file:
                                    output_file.write(command + "\n")  # Save the command for reference
    
                                    # Process output line by line
                                    for line in process.stdout:
                                        # Check if the line starts with #TRAIN# or #TEST#
                                        if line.startswith("#TRAIN#") or line.startswith("#TEST#"):
                                            output_file.write(line)  # Write matching lines to the file
    
                                # Wait for the process to complete
                                process.communicate()
    
                                print("Command executed, and filtered output saved to output_ht_0912.txt")