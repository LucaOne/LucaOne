# LucaOne(LucaGPLM)     
LucaOne: Generalized Biological Foundation Model with Unified Nucleic Acid and Protein Language.  

## 1. LucaOne Workflow      

<center>
<img alt="The workflow of LucaOne." src="./pics/ModelArchitecture.png"/>

Fig. 1 The workflow of LucaOne.   
</center>   


## 2. LucaOne PreTraining Data & PreTraining Tasks

<center>
<img alt="The data and tasks for pre-training LucaOne, and T-SNE on four embedding models." src="./pics/PretraingData&Tasks&EmbeddingTSNE.png"/>

Fig. 2 The data and tasks for pre-training LucaOne, and T-SNE on four embedding models.     
</center>


## 3. Downstream Tasks

<center>
<img alt="Downstream task network with three input types and results comparison of 8 verification tasks." src="./pics/DownstreamNetworksAndMetrics.png"/>

Fig. 3 Downstream task network with three input types and results comparison of 8 verification tasks.  
</center>


## 4. Environment Installation
### step1: update git
#### 1) centos
sudo yum update     
sudo yum install git-all

#### 2) ubuntu
sudo apt-get update     
sudo apt install git-all

### step2: install python 3.9
#### 1) download anaconda3
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

#### 2) install conda
sh Anaconda3-2022.05-Linux-x86_64.sh
##### Notice: Select Yes to update ~/.bashrc
source ~/.bashrc

#### 3) create a virtual environment: python=3.9.13
conda create -n lucaone python=3.9.13


#### 4) activate lucaone
conda activate lucaone

### step3:  install other requirements
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple   


## 5. Inference   
You can use the project: **<a href='https://github.com/LucaOne/LucaOneApp'>LucaOneApp Github</a> or <a href='http://47.93.21.181/lucaone/LucaOneApp'>LucaOneApp FTP</a>** for **embedding inference**. For details, please refer to the **`README`** of the LucaOneApp project.        

The project will download automatically LucaOne Trained-CheckPoint from **FTP**.     

## 6. For Downstream Tasks    
This project: **<a href='https://github.com/LucaOne/LucaOneTasks'>LucaOneTasks Github</a> or <a href='http://47.93.21.181/lucaone/LucaOneTasks'>LucaOneTasks FTP</a>** is all the downstream tasks used in our paper(**based on LucaOne's Embedding**), and you can use this project to run other tasks, please refer to the **`README`** of this project. 

## 7. Dataset   
Pretraining Dataset FTP: <a href='http://47.93.21.181/lucaone/PreTrainingDataset/dataset/lucagplm'>Dataset for LucaOne</a>     

Copy the dataset from <href> http://47.93.21.181/lucaone/PreTrainingDataset/dataset/lucagplm </href> into the directory: `./dataset/` 
 
The training dataset(`dataset/lucagplm/v2.0/train/`) whose file names start with **'2023112418163521'** are gene data(DNA + RNA), and those that start with **'2023112314061479'** are protein data.

The validation dataset(`dataset/lucagplm/v2.0/dev/`) whose file names start with **'2023112418224620'** are gene data(DNA + RNA), and those that start with **'2023112314080544'** are protein data.

The testing dataset(`dataset/lucagplm/v2.0/test/`) whose file names start with **'2023112418231445'** are gene data(DNA + RNA), and those that start with **'2023112314083364'** are protein data.

**Notice**     
If you want to train individual nucleic acid or protein LucaOne(LucaOne-Gene or LucaOne-Prot), please separate the datasets as described above.   

## 8. Training Scripts   
Training scripts are under the directory `src/training`, including 4 shell scripts:    
`run_multi_v2.0.sh`:  nucleic acid(DNA+RNA) and protein mixed training with 10 pre-training tasks.   
`run_multi_mask_v2.0.sh`:  nucleic acid(DNA+RNA) and protein mixed training with only 2 mask pre-training tasks.       
`run_multi_v2.0_gene.sh`:  individual nucleic acid training with 3 pre-training tasks.   
`run_multi_v2.0_prot.sh`:  individual protein training with 7 pre-training tasks.    

## 9. Data and Code Availability     
**FTP:**   
Pre-training data, code, and trained checkpoint of LucaOne, embedding inference code, downstream validation tasks data & code, and other materials are available: <a href='http://47.93.21.181/lucaone/'>FTP</a>. 

**Details:**     

The LucaOne's model code is available at: <a href='https://github.com/LucaOne/LucaOne'>LucaOne Github </a> or <a href='http://47.93.21.181/lucaone/LucaOne/'>LucaOne</a>.   

The trained-checkpoint files are available at: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/'>TrainedCheckPoint</a>.  

LucaOne's representational inference code is available at: <a href='https://github.com/LucaOne/LucaOneApp'>LucaOneApp Github</a> or <a href='http://47.93.21.181/lucaone/LucaOneApp'>LucaOneApp</a>. 

The project of 8 downstream tasks is available at: <a href='https://github.com/LucaOne/LucaOneTasks'>LucaOneTasks Github</a> or <a href='http://47.93.21.181/lucaone/LucaOneTasks'>LucaOneTasks</a>.

The pre-training dataset of LucaOne is opened at: <a href='http://47.93.21.181/lucaone/PreTrainingDataset/'>PreTrainingDataset</a>. 

The datasets of downstream tasks are available at: <a href='http://47.93.21.181/lucaone/DownstreamTasksDataset/'> DownstreamTasksDataset </a>. 

Other supplementary materials are available at: <a href='http://47.93.21.181/lucaone/Others/'> Others </a>.


## 10. Contributor        
<a href="https://scholar.google.com.hk/citations?user=RDbqGTcAAAAJ&hl=en" title="Yong He">Yong He</a>, 
<a href="https://scholar.google.com/citations?user=lT3nelQAAAAJ&hl=en" title="Zhaorong Li">Zhaorong Li</a>, 
<a href="https://scholar.google.com/citations?user=ODcOX4AAAAAJ&hl=zh-CN" title="Pan Fang">Pan Fang</a>,
<a href="https://scholar.google.com/citations?view_op=list_works&hl=en&user=uvrzUfEAAAAJ" title="Yongtao Shan">Yongtao Shan</a>, Yanhong Wei, 
<a href="https://scholar.google.com.hk/citations?hl=zh-CN&pli=1&user=Zhlg9QkAAAAJ" title="Yuan-Fei Pan">Yuan-Fei Pan</a>

## 11. Citation          
@article {LucaOne,                
author = {Yong He and Pan Fang and Yongtao Shan and Yuanfei Pan and Yanhong Wei and Yichang Chen and Yihao Chen and Yi Liu and Zhenyu Zeng and Zhan Zhou and Feng Zhu and Edward C. Holmes and Jieping Ye and Jun Li and Yuelong Shu and Mang Shi and Zhaorong Li},     
title = {LucaOne: Generalized Biological Foundation Model with Unified Nucleic Acid and Protein Language},      
elocation-id = {2024.05.10.592927},        
year = {2024},         
doi = {10.1101/2024.05.10.592927},        
publisher = {Cold Spring Harbor Laboratory},        
URL = {https://www.biorxiv.org/content/early/2024/05/14/2024.05.10.592927},        
eprint = {https://www.biorxiv.org/content/early/2024/05/14/2024.05.10.592927.full.pdf},        
journal = {bioRxiv}        
}       







