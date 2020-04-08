FROM gitlab-registry.nautilus.optiputer.net/prp/jupyterlab:latest

LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER $NB_USER

RUN set -x \
    && pip install coffea tables mplhep pyjet
    

# RUN git clone https://gitlab.nautilus.optiputer.net/jmduarte/anomalydetection4jets.git $HOME/AnomalyDetection4Jets && fix-permissions $HOME/AnomalyDetection4Jets 
# WORKDIR $HOME/AnomalyDetection4Jets
# ADD https://zenodo.org/record/3596919/files/events_LHCO2020_backgroundMC_Pythia.h5?download=1 events_LHCO2020_backgroundMC_Pythia.h5
# ADD https://zenodo.org/record/3596919/files/events_LHCO2020_BlackBox1.h5?download=1 events_LHCO2020_BlackBox1.h5
# ADD https://zenodo.org/record/3596919/files/events_LHCO2020_BlackBox2.h5?download=1 events_LHCO2020_BlackBox2.h5
# ADD https://zenodo.org/record/3596919/files/events_LHCO2020_BlackBox3.h5?download=1 events_LHCO2020_BlackBox3.h5
