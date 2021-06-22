# Particle Graph Autoencoders for Anomaly Detection
![GAE_new-1](https://user-images.githubusercontent.com/42155174/98452299-ca24e980-2102-11eb-9474-c67a67f31923.png)

## Project Description
This project implements autoencoders with graph neural networks using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for application in anomaly detection in particle collisions at the Large Hadron Collider.

An autoencoder trained to reconstruct data can be used to filter out background from potentially anomalous signals by cutting on the reconstruction loss. We can then analyze the filtered data using a bump hunt. More details can be found in this community submission of our preliminary results (DL link): [GraphAutoencoderLHCO2020PaperContribution.pdf](https://github.com/stsan9/AnomalyDetection4Jets/files/5505568/GraphAutoencoderLHCO2020PaperContribution.pdf). Also section 3.7 in the [community paper on arXiv](https://arxiv.org/pdf/2101.08320.pdf).

## Dataset
Dataset comes from the [LHC Olympics 2020 Anomaly Detection Challenge](https://lhco2020.github.io/homepage/). Dataset and details can be found at the following:

[Background Data and Black Boxes](https://zenodo.org/record/3596919#.XkSGTRNKhTZ)

[R&D Dataset](https://zenodo.org/record/2629073#.XKdewGXlRg0)

# PRP Instructions
Follow [Nautilus instructions](https://ucsd-prp.gitlab.io/userdocs/start/toc-start/) to get access to [PRP](https://ucsd-prp.gitlab.io/) under the cms-ml namespace and set up kubectl on your computer

Fork the repo and clone it locally
```
git clone [the URL of your fork]
```

In the directory where `anomaly-pod.yml` is, run
```
kubectl -n cms-ml create -f anomaly-pod.yml
kubectl -n cms-ml exec -it anom-pod -- bash
```
This lets you access a pod (remote environment) where our data and models are stored.

To work on the code within the pod and edit with Vim or Emacs:
```
cd ~/work
git clone [the URL of your fork]
cd AnomalyDetection4Jets/code
```

To see what's stored in our volume, do `cd /anomalyvol/` and look around the directory. Important directories include:
- `/anomalyvol/experiments`
- `/anomalyvol/data`
- `/anomalyvol/emd_models`

Once done using your pod terminate it with:
```
kubectl -n cms-ml delete pods anom-pod
```

### Notes
- It's mandatory to read the [Nautilus documentation](https://ucsd-prp.gitlab.io/userdocs/start/toc-start/) for usage policies and other details
- There are 2 volumes where we store our data, you can see them if you run `kubect -n cms-ml get pvc` in your local environment.
  - `anomalyvol-2` is more up to date but if you need extra storage or if you want space to experiment more you can switch to using `anomalyvol` by changing any instances of `anomalyvol-2` to `anomalyvol` in the .yml files.
```
------------------------
anomalyvol              Bound    pvc-a5cb2fae-e8e3-4c0d-b8ef-e69ff52f5aad   1000Gi     RWX            rook-cephfs        357d
anomalyvol-2            Bound    pvc-6bb604aa-5a40-44d4-af1e-fb6f38b4d1fb   1000Gi     RWX            rook-cephfs        301d
...
```
- Pods only have a lifespan of 6hrs so save your work frequently by pushing to github. Alternatively edit the code on your local machine instead of through a pod if you don't need to run any commands. It can be convenient to generate a small sample of the dataset on your local machine to test your code after you set up all the packages needed locally.
- You can set the default namespace to be `cms-ml` so that you don't have to always add the `-n cms-ml` flag to kubectl commands.
```
kubectl config set-context nautilus --namespace=cms-ml
```
-Any changes you save in the home directory of a pod will get deleted (~) once the pod terminates. Things saved in `/anomalyvol/` will remain.

# Running the Code
To know how to run the code and what all the flags do I recommend looking through the argparse section of the corresponding files. Below are some examples of how the commands will look.

## Generate the Dataset
Make sure you have a directory somewhere with the raw data in the `raw/` directory. Look at `/anomalyvol/data/bb_train_sets/bb0_xyz/` for reference. The raw data can be downloaded from the [Zenodo page](https://zenodo.org/record/3596919#.XkSGTRNKhTZ) linked above, and sent to the volume using:
```
kubectl -n cms-ml cp events_LHCO2020_backgroundMC_Pythia.h5 cms-ml/anom-pod:/anomalyvol/data/bb_train_sets/your_directory/raw/events_LHCO2020_backgroundMC_Pythia.h5
```
Replace the name of the file or the path with whatever you're sending over.

To process a sample of the raw dataset do
```
python graph_data.py --dataset /anomalyvol/data/[examplepath] --n-proc 1 --n-events 1000 --bb 0 --n-events-merge 100
```
Usually creating the whole dataset takes a long time and memory so we'll use a job instead. You can check `anomaly-graph-job.yml` for details on how that would look. Assuming all the parameters are correctly set and the dataset exists, run the job with:
```
kubectl -n cms-ml create -f anomaly-graph-job.yml
```
You can delete the job once using:
```
kubectl -n cms-ml delete jobs anomaly-graph-job.yml
```

Notes:
- you do not want to generate the whole dataset on your local machine. Once processed it takes a lot of space.
- for bb0 (the background dataset we use for training) you can find it preprocessed in `/anomalyvol/data/bb_train_sets/bb0_xyz/`

## Training
```
python train_script.py --mod-name [REPLACE WITH A NAME] --input-dir /anomalyvol/data/bb_train_sets/bb0_xyz/ --box-num 0 --model EdgeNet --batch-size 16 --lr 0.01 --loss emd_loss --emd-model-name EmdNNRel.best.pth --num-data 256 --patience 10
```
You can find the saved model in `/anomalyvol/experiments/` in the directory with the corresponding model name. Alternatively you can change the output path using the `--output-dir` flag.
You won't have enough memory to train a model using the whole dataset in a pod, so set up a train job with the appropriate parameters. Check `gae_train_job.yml` for an example.

If your pod or job has multiple gpus allocated to it, it will by default use multiple gpus for training.
If using the emd network as a loss check out `/anomalyvol/emd_models/` for their names (I only recommend using the ones suffixed with Spl and Rel). If using multi-gpus to train the emd network, set `--model` to `EdgeNetEMD`. 

## Bump Hunt
Same idea as the prior sections, though you will almost always want to run a job. Look at `bump_hunt.py` for the flags. By default you will find the generated graphs in `/anomalyvol/experiments`.
