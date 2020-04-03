FROM gitlab-registry.nautilus.optiputer.net/prp/jupyterlab:latest

LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

RUN set -x \
    && pip install coffea tables mplhep

RUN git clone https://github.com/stsan9/AnomalyDetection4Jets.git .

ADD events_LHCO2020_backgroundMC_Pythia.h5
ADD events_LHCO2020_BlackBox1.h5
ADD events_LHCO2020_BlackBox2.h5
ADD events_LHCO2020_BlackBox3.h5