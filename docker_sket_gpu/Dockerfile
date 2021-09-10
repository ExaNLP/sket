# set nvidia version
FROM nvidia/cuda:11.0-devel

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3.8
RUN apt-get -y install python3-pip

# set work directory
WORKDIR /sket
# copy requirements in work directory
COPY requirements.txt /sket
# install requirements and scispacy model
RUN pip install --no-cache-dir --user -r requirements.txt
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_sm-0.3.0.tar.gz

# copy code and config files within work directory
COPY . /sket
# run sket
ENTRYPOINT ["python3", "./run_med_sket.py"]
