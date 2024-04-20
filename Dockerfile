from python:3.9.10

# set up the working dir
WORKDIR /app/src

# create a new environment
RUN apt update
RUN conda create -n py39 python=3.9 pip
RUN echo "source activate py39" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

# COPY code and dependencies required by the pipeline
COPY requirements.txt .

# install dependencies
RUN pip install --upgrade pip && \
    pip install setuptools && \
    pip install\
    -r requirements.txt

COPY . .

CMD ["python", "pipline.py"]
