# Wasserstein Normalized AutoEncoder (WNAE)

This branch contains an example setup to run a WNAE training based on the [SVJanalysis repo](https://github.com/eth-svj/SVJanalysis).    


## Installation

```
git clone git@github.com:fleble/wnae.git
```
The needed packages and versions can be found in `requirements.txt`.    

If you want to run the example setup in this branch, you will need more pakcages, refer to the documentation of the [SVJanalysis repo](https://github.com/eth-svj/SVJanalysis).    


## Usage

This repo provides:
* the WNAE implementation
* an example data loader
* an example encoder / decoder architecture
* an example training loop

The file `run_training.py` can be used to run trainings and shows how the different pieces talk to each other.    