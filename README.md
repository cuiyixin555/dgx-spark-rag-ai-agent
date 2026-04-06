# This is analysis tool based on RAG with NVIDIA API
2026/04/06

Author: CUI Xin, CUI Yixin & CUI Xiaofen

# Overview:

This is a RAG intelligent paper analysis tool based on the NVIDIA API. The UI is based on Gradio's automatically generated web page. The tool can support the analysis of pdf, png, jpeg and txt multiple file formats. The visual models used include microsoft/phi-3.5-vision-instruct, meta/llama-3.2-11b-vision-instruct. The word embedding models used include nvidia/nv-embedqa-e5-v5. The above model is lightweight and fast, enabling our analysis tools to respond quickly.

# How to setup SPARK DGX environment
In mainland China, you need to switch to the Tsinghua University mirror source.

## Step1: Creating a Python 3.13 virtual environment
conda create --name analysis_rag python=3.13

## Step2: Entering a virtual environment
conda activate analysis_rag

## Step3: Install the various modules required for RAG
pip install langchain-nvidia-ai-endpoints

pip install jupyterlab

pip install langchain_core

pip install langchain

pip install langchain_community

pip install matplotlib

pip install numpy

pip install faiss-cpu

pip install openai

# Step4 Run 
python analysis.py


# Reference

1\. NVIDIA Build： https://build.nvidia.com/

2\. NVIDIA DLI：<https://www.nvidia.cn/training/online/>

3\. <https://github.com/kinfey/Microsoft-Phi-3-NvidiaNIMWorkshop>

​
