FROM gitlab-registry.nautilus.optiputer.net/prp/jupyterlab:latest

LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    NB_USER=jovyan \
    NB_UID=1000 \
    NB_GID=100 \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV PATH=$CONDA_DIR/bin:$PATH \
    HOME=/home/$NB_USER

USER $NB_USER

RUN set -x \
    && pip install coffea tables mplhep
    
USER $NB_USER

WORKDIR $HOME/AnomalyDetection4Jets
RUN git clone https://gitlab.nautilus.optiputer.net/jmduarte/anomalydetection4jets.git .
ADD https://zenodo.org/record/3596919/files/events_LHCO2020_backgroundMC_Pythia.h5?download=1 events_LHCO2020_backgroundMC_Pythia.h5
ADD https://zenodo.org/record/3596919/files/events_LHCO2020_BlackBox1.h5?download=1 events_LHCO2020_BlackBox1.h5
ADD https://zenodo.org/record/3596919/files/events_LHCO2020_BlackBox2.h5?download=1 events_LHCO2020_BlackBox2.h5
ADD https://zenodo.org/record/3596919/files/events_LHCO2020_BlackBox3.h5?download=1 events_LHCO2020_BlackBox3.h5