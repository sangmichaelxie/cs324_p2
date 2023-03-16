FROM codalab/default-gpu
SHELL ["/bin/bash", "-c"]
LABEL maintainer="xie@cs.stanford.edu"

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

RUN git clone git://github.com/yyuu/pyenv.git .pyenv

ENV HOME /
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN pyenv install 3.8.5
RUN pyenv global 3.8.5
RUN pyenv rehash
RUN pip install --upgrade pip

RUN pip install -U --no-cache-dir \
      numpy \
      scipy \
      matplotlib \
      pandas \
      sympy \
      nose \
      spacy \
      tqdm \
      wheel \
      scikit-learn \
      nltk \
      tensorboard
RUN python -m spacy download en_core_web_sm

RUN pip install --no-cache-dir \
      torch==1.8.1+cu111 \
      -f https://download.pytorch.org/whl/torch_stable.html

# Install apex
WORKDIR /tmp/apex_installation
RUN git clone https://github.com/NVIDIA/apex.git && cd apex && git checkout 0c2c6eea6556b208d1a8711197efc94899e754e1
WORKDIR /tmp/apex_installation/apex
RUN pip install --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
WORKDIR /

ADD ./requirements_docker.txt requirements.txt
RUN pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt')"
