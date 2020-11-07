# Particle Graph Autoencoders for Anomaly Detection
![GAE_new-1](https://user-images.githubusercontent.com/42155174/98452299-ca24e980-2102-11eb-9474-c67a67f31923.png)

## Project Description
This project implements autoencoders with graph neural networks using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for application in anomaly detection in particle collisions at the Large Hadron Collider.

An autoencoder trained to reconstruct data can be used to filter out background from potentially anomalous signals by cutting on the reconstruction loss. We can then analyze the filtered data using a bump hunt. More details can be found in this community submission of our preliminary results: [GraphAutoencoderLHCO2020PaperContribution.pdf](https://github.com/stsan9/AnomalyDetection4Jets/files/5505568/GraphAutoencoderLHCO2020PaperContribution.pdf).

## Dataset
Dataset comes from the [LHC Olympics 2020 Anomaly Detection Challenge](https://lhco2020.github.io/homepage/). Dataset and details can be found at the following:

[Background Data and Black Boxes](https://zenodo.org/record/3596919#.XkSGTRNKhTZ)

[R&D Dataset](https://zenodo.org/record/2629073#.XKdewGXlRg0)

## WIP:
- [ ] Normalizing Flow
- [ ] Train directly on R&D dataset with different amounts of injected signal and generate ROC curves
- [ ] Speed up model training
- [ ] Experiment with Hungarian Loss
