FROM python:3.8-buster
ENV PYTHONUNBUFFERED=1
WORKDIR /code
COPY ./docker-sket_server-config/requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_sm-0.3.0.tar.gz
COPY . /code/

