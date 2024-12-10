# CEVAE_project

Content description-
 
.
 model/                  Contains implementations of various VAE architectures <br>
 model/hvae_mu.py        Contains implementation of Hierarchical VAE architecture. <br>
 model/vqvae.py          Contain implementation of VQ-VAE architecture. <br>
 a1.py                   Script for running different model configurations and hyperparameters <br>
 parallel_script.py      Runs multiple scripts concurrently <br>
 results_syn.py          Reproduces Figure 3 from the research paper <br>
 results_beta.py         Reproduces Figure 3 with different beta values for the beta-VAE model <br>
 result_beta_t1.py       Reproduces Table 1 & Table 2 for beta-VAE model configurations <br>


Note: Different networks.py file (networks_hm.py) was created to modify encoder architecture according to Hierarchical VAE. 
 
## Steps for replicating results-

To generate Table 2,3,4 and 5 values for beta-VAE architecture with different beta values, run result_beta_t1.py script. A new text file results_scores_beta_t1.txt will be generated and stored in current directory having all the results.

To generate Figure 2, run result_beta.py file, all the results will be stored in results_scores_beta.txt file generated through python script. 
After that run figure_2_rep.ipynb to generate final image, where Test ATE results obtained from different models are stored in seperate list. 
