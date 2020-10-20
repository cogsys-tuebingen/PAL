Run and adapt ./code_/main.py to reproduce our experiments.
-> Since this evaluation is done with outdated Tensorflow 1.5 we refer to https://github.com/haven-ai/haven-ai if you are searching for a good framework to compare optimizers. 
-> Note, that the network to train on has to be decommented in the code of ./code_/main.py.
-> Note, that we used a slurm scheduler based cluster to run our experiments.
Thus, the main.py script is called in our experiments by multiple other files conducting grid searches, but those won't work if you do not have an identical slurm configuration. Therefore, they are not included.
