FROM pytorch/pytorch:latest

RUN apt-get update
RUN apt-get -y install git

RUN pip install --upgrade pip
RUN pip install jupyterlab

EXPOSE 9005
# 2022/5/17作成　山口さんのjupyter_notebookを使用するため
