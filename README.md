# DRAM
source code for the paper "DRAM: Dynamic Rule-Aware Memory for Knowledge Graph Reasoning"
# Installation
Create a virtual environment
 ```bash
 conda create --name DRAM python=3.9 -y
 ```
Install PyTorch 
 ```bash
 conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
Install other Python libraries
```bash
 pip install scipy easydict tqdm matplotlib 
```
# How to Run
The folder contain the configuration file of each dataset and the folder provides the implementation of DRAM. You can edit the config file and enter the folder to excute the following command:
```bash
 python main.py --config ../configs/umls.json
```
