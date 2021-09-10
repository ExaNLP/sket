# set python 3.8 as python version
FROM python:3.8-slim

# install OS packages
RUN apt update && apt install --no-install-recommends -y build-essential gcc && apt clean && rm -rf /var/lib/apt/lists/*

# set work directory
WORKDIR /sket
COPY requirements.txt /sket

# install requirements and scispacy model
RUN pip install --no-cache-dir --user -r requirements.txt
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_sm-0.3.0.tar.gz

# copy code and config files within work directory
COPY . /sket

# run sket
ENTRYPOINT ["python", "./run_med_sket.py"]
