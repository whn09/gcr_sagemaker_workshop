# ==================================================================
# module list
# ------------------------------------------------------------------
# python        2.7    (apt)
# tensorflow    latest (pip)
# ==================================================================

FROM nvidia/cuda:8.0-cudnn6-devel
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="pip --no-cache-dir install " && \
    GIT_CLONE="git clone --depth 10" && \

    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    #apt-get update && \
    apt-get update && apt-get install -y --allow-unauthenticated \
                        python-tk \
                        python-opencv \
                        nginx \
                        && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
        && \

# ==================================================================
# python
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python-pip \
        python-dev \
        && \
    $PIP_INSTALL \
        setuptools \
        pip \
        && \
    $PIP_INSTALL \
        numpy==1.15.4 \
        scipy==1.2.0 \
        pandas==0.23.4 \
        scikit-learn==0.20.0 \
        matplotlib==2.2.3 \
        Cython==0.24 \
        easydict==1.9 \
        opencv-python==4.1.0.25 \
        h5py==2.9.0 \
        PyYAML==5.1.2 \
        flask \
        gevent \
        gunicorn \
        Pillow \
        boto3 \
         && \

# ==================================================================
# tensorflow
# ------------------------------------------------------------------

    $PIP_INSTALL \
        tensorflow_gpu \
        && \


# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY ctpn /opt/program
COPY checkpoints /opt/ml/checkpoints/
WORKDIR /opt/program