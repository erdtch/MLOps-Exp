FROM python:3

RUN apt-get update && apt-get install make
RUN apt install build-essential -y --no-install-recommends

WORKDIR /home/darknet
RUN git clone https://github.com/AlexeyAB/darknet

#RUN sed -i 's/OPENCV=0/OPENCV=1/' Makefile
RUN make

RUN pip install --upgrade jupyter

# jupyter notebook
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]