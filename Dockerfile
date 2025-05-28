FROM python:3.11-slim

# set non-interactive frontend for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libhdf5-dev \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# upgrade pip and install pip tools
RUN pip install --upgrade pip

# cd to home
WORKDIR /home/pontus

# copy the rest of the source and install it 
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# set working directory
WORKDIR /home

# default command for container
CMD ["/bin/bash"]










