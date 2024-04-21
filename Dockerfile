from python:3.9.10

# set up the working dir
WORKDIR /app/src

# COPY code and dependencies required by the pipeline
COPY requirements.txt .

# install dependencies
RUN pip install --upgrade pip && \
    pip install setuptools && \
    pip install\
    -r requirements.txt

COPY . .

CMD ["python" ,"pipeline.py"]
