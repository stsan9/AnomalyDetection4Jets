FROM gitlab-registry.nautilus.optiputer.net/prp/jupyterlab:latest

LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

RUN set -x \
    && pip install coffea tables mplhep

WORKDIR /AnomalyDetection4Jets
RUN git clone https://github.com/stsan9/AnomalyDetection4Jets.git .
ADD https://zenodo.org/record/3596919/files/events_LHCO2020_backgroundMC_Pythia.h5?download=1 events_LHCO2020_backgroundMC_Pythia.h5
ADD https://zenodo.org/record/3596919/files/events_LHCO2020_BlackBox1.h5?download=1 events_LHCO2020_BlackBox1.h5
ADD https://zenodo.org/record/3596919/files/events_LHCO2020_BlackBox2.h5?download=1 events_LHCO2020_BlackBox2.h5
ADD https://zenodo.org/record/3596919/files/events_LHCO2020_BlackBox3.h5?download=1 events_LHCO2020_BlackBox3.h5