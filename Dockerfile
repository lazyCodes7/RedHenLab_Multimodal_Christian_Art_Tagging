FROM python
COPY ./ ./
RUN apt-get update && apt-get -y install gcc
RUN apt-get -y install libgl1
RUN pip install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt
