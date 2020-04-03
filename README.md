# AnomalyDetection4Jets

download files from zenodo
```
wget https://zenodo.org/record/3596919/files/events_LHCO2020_backgroundMC_Pythia.h5?download=1 -O events_LHCO2020_backgroundMC_Pythia.h5
wget https://zenodo.org/record/3596919/files/events_LHCO2020_BlackBox1.h5?download=1 -O events_LHCO2020_BlackBox1.h5
wget https://zenodo.org/record/3596919/files/events_LHCO2020_BlackBox2.h5?download=1 -O events_LHCO2020_BlackBox2.h5 
wget https://zenodo.org/record/3596919/files/events_LHCO2020_BlackBox3.h5?download=1 -O events_LHCO2020_BlackBox3.h5
```

create environment (once)
```
conda env create -f environment.yml
```

set up environment (each time)
```
conda activate anomaly
```
