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

Table 2,3,4&5 for different bvae architectures-
