# DRAM
source code for the paper "DRAM: Dynamic Rule-Aware Memory for Knowledge Graph Reasoning"
# Installation
Create a virtual environment
<pre> 
 conda create --name DRAM python=3.9 -y
</pre>
Install PyTorch 
<pre> 
 conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
</pre>
Install other Python libraries
<pre> 
 pip install scipy easydict tqdm matplotlib 
</pre>
# How to Run
The folder contain the configuration file of each dataset and the folder provides the implementation of DRAM. You can edit the config file and enter the folder to excute the following command:
<pre> 
 python main.py --config ../configs/umls.json
</pre>
