# From the Python container
FROM python:slim

# Set working directory
WORKDIR /app

# Copy the environment
COPY environment.yml .

RUN apt-get update && apt-get -y upgrade \
    && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    gcc \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda env create -f environment.yml && \
    conda install python=3.8 pip

# create the environment
# RUN conda install -c conda-forge fbprophet
# RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "crypto", "/bin/bash", "-c"]

# Copy the garbage
COPY src/ ./src
COPY assets/ ./assets
COPY main.py .
COPY .gitignore .
COPY README.md .
COPY LICENSE .

# run main
ENTRYPOINT ["conda", "run", "-n", "crypto", "python", "./main.py"]