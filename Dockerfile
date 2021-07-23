FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp:latest

LABEL maintainer="Steven Tsan <stsan@ucsd.edu>"

USER $NB_USER

ENV USER=${NB_USER}

RUN sudo apt-get update
RUN sudo apt-get install -y vim


RUN set -x \
    && pip install coffea tables mplhep pyjet llvmlite --ignore-installed \
    && pip install git+git://github.com/jmduarte/pyBumpHunter.git@cosmetics#egg=pyBumpHunter \
    && pip install pot energyflow \
    && pip install natsort \
    && pip install qpth cvxpy

ENV TORCH=1.8.1
ENV CUDA=cu102
    
RUN set -x \
    && pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html \
    && pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html \
    && pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html \
    && pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html \
    && pip install torch-geometric

RUN set -x \
    && fix-permissions /home/$NB_USER
