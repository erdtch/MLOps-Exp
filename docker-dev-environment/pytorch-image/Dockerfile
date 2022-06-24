FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

WORKDIR /home/pytorch

COPY . .

RUN pip install -r requirements.txt
RUN pip install jupyterlab

# jupyter notebook
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]