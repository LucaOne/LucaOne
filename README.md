# LucaGroup  
Pioneering research of AI for Life Science at **Alibaba Cloud**.    
Our work focuses on building foundational models to decode the language of life.    

📚 Publications
* **LucaProt**    
<a href="https://www.cell.com/cell/fulltext/s0092-8674(24)01085-7">"Using artificial intelligence to document the hidden RNA virosphere."</a> **Cell (Cover Article), 2024**.   

* **LucaOne**     
<a href="https://www.nature.com/articles/s42256-025-01044-4">"Generalized biological foundation model with unified nucleic acid and protein language."</a> **Nature Machine Intelligence, 2025**.   

* **LucaVirus**    
<a href="https://www.biorxiv.org/content/10.1101/2025.06.14.659722v1">"Predicting the evolutionary and functional landscapes of viruses with a unified nucleotide-protein language model: LucaVirus."</a> **Biorxiv, 2025**.

* **LucaPCycle**       
<a href="https://www.nature.com/articles/s41467-025-60142-4">"Illuminating microbial phosphorus cycling in deep-sea cold seep sediments using protein language models."</a> **Nature Communications, 2025**.

* **LucaAMANet**    
<a href="https://dl.acm.org/doi/abs/10.1145/3394486.3403055">"Attention and memory-augmented networks for dual-view sequential learning."</a> **SIGKDD, 2020**.    

  
🏆 Awards & Competitions       
* **"Top 10 Progress in Bioinformatics in China (2024)"** - LucaProt.   
* **1st Place**, Clinical Automated Coding Task, CHIP 2022.     

💡 Patents & Innovation      
* **Patent Portfolio:**     
  * **15** patent applications filed
  * **10** patents granted
  * **1** PCT international patent pending
  
* **Industry Awards:**   
  * **"Top 10 Patent"** Award, Alibaba Cloud & Tongyi Lab (FY2025)
  * **3rd Prize Patent** Award, Alibaba Cloud (FY2018)   

# LucaOne(LucaGPLM)     
LucaOne: Generalized Biological Foundation Model with Unified Nucleic Acid and Protein Language.

# TimeLine         
* **2025/12/31**    
LucaOne now supports the Hugging Face interface for further training.      
It allows for various training modes, including using sequence-only inputs or injecting biological knowledge following the LucaOne framework. You can fine-tune the model for both sequence-level and token-level classification or regression tasks.        
Please refer to the Hugging Face address: https://huggingface.co/collections/LucaGroup/lucaone, or the `huggingface` branch of this repository.

  - **Hugging Face Native**: Full support for `AutoModel`, `AutoModelForMaskedLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`, `AutoConfig`, and `AutoTokenizer`.
  - **Unified Architecture**: Single model architecture handling multiple biological modalities.
  - **Task-Specific Heads**:
      - `LucaGPLMModel`: For sequences embedding.
      - `LucaGPLMForMaskedLM`: For pre-training and sequence recovery.
      - `LucaGPLMForSequenceClassification`: For sequence-level tasks (e.g., protein family, solubility, or promoter prediction).
      - `LucaGPLMForTokenClassification`: For residue-level tasks (e.g., secondary structure, binding sites, or post-translational modifications).
  - **Extensible**: Easily adaptable to custom downstream tasks using the standard `transformers` API.


* 2025/12/26:   
LucaOne now supports **BF16** for training and embedding inference.    
add parameter: **--use_bf16**     

* 2025/08/15:   
**Huggingface**     
<a href='https://huggingface.co/LucaGroup'>https://huggingface.co/LucaGroup </a>     

* 2025/06/16:  
**The Pretraining Dataset uploaded CNSA**      
The pre-training dataset of LucaOne has been deposited into CNGB Sequence Archive (CNSA) with accession number CNP0007266 (https://db.cngb.org/search/project/CNP0007266/).   


* 2025/04/08:
  * **LucaOne**          
    add `checkpoint=36000000` for `LucaOne`     
    location: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/latest/models/lucaone/lucaone/checkpoint-step36000000/'>checkpoint-step36000000</a>
  * **LucaOne-Gene**          
    add `checkpoint=36800000` for `LucaOne-Gene` (only trained using `DNA` and `RNA`)     
    location: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/latest/models/lucaone/lucaone-gene/checkpoint-step36800000/'>checkpoint-step36800000</a>
  * **LucaOne-Prot**       
    add `checkpoint=30000000` for `LucaOne-Prot` (only trained using `Protein`)       
    location: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/latest/models/lucaone/lucaone-prot/checkpoint-step30000000/'>checkpoint-step30000000</a>

* 2024/10/01: optimized embedding inference code: `src/get_embedding.py`      
* 2024/08/01: add `checkpoint=17600000`, location: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step17600000/'>checkpoint-step17600000</a>   
* 2024/07/24: feature: add `continue training when failure`   

This project will download the checkpoint automatically from our `FTP` according to the value of parameter:
* **--llm_type**
* **--llm_version**
* **--llm_step**


## Embedding Recommendation
| --llm_type | --llm_version  |              --llm_step              |                 Usage (seq_type)                 |
|:----------:|:--------------:|:------------------------------------:|:------------------------------------------------:|
| `lucaone`  |   `lucaone`    | `36000000`, `17600000`, or `5600000` | both `gene` (i.e. `DNA`, `RNA`) and `prot` sequences |
| `lucaone`  | `lucaone-gene` |              `36800000`              |    only for `gene` (i.e. `DNA`, `RNA`) sequences     |
| `lucaone`  | `lucaone-prot` |              `30000000`              |             only for `prot` sequence             |


## 1. LucaOne Workflow      

<center>
<img alt="The workflow of LucaOne." src="./pics/ModelArchitecture.jpg"/>

Fig. 1 The workflow of LucaOne.   
</center>   


## 2. LucaOne PreTraining Data & PreTraining Tasks & Embedding(Zero-Shot)      

<center>
<img alt="The data and tasks for pre-training LucaOne, and T-SNE on four embedding models." src="./pics/PretraingData&Tasks&EmbeddingTSNE.png"/>

Fig. 2 The data and tasks for pre-training LucaOne, and T-SNE on four embedding models.     
</center>

## 3. Central Dogma(Few-Shot)    

<center>
<img alt="Learning Central Dogma of Molecular Biology." src="./pics/CentralDogma.png"/>

Fig. 3 Learning Central Dogma of Molecular Biology.
</center>

## 4. Downstream Tasks(SFT)    

<center>
<img alt="Downstream task network with three input types and results comparison of 8 verification tasks." src="./pics/DownstreamNetworksAndMetrics.png"/>

Fig. 4 Downstream task network with three input types and results comparison of 8 verification tasks.  
</center>


## 5. Environment Installation
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


## 6. Embedding Inference   
You can use the project: **<a href='https://github.com/LucaOne/LucaOneApp'>LucaOneApp Github</a> or <a href='http://47.93.21.181/lucaone/LucaOneApp'>LucaOneApp FTP</a>** for **embedding inference**. For details, please refer to the **`README`** of the LucaOneApp project.        

The project will download automatically LucaOne Trained-CheckPoint from **FTP**.     

or code in 'src/get_embedding.py'   

## 7. For Downstream Tasks    
This project: **<a href='https://github.com/LucaOne/LucaOneTasks'>LucaOneTasks Github</a> or <a href='http://47.93.21.181/lucaone/LucaOneTasks'>LucaOneTasks FTP</a>** is all the downstream tasks used in our paper(**based on LucaOne's Embedding**), and you can use this project to run other tasks, please refer to the **`README`** of this project. 

## 8. Dataset   
Pretraining Dataset FTP: <a href='http://47.93.21.181/lucaone/PreTrainingDataset/dataset/lucagplm'>Dataset for LucaOne</a>     

Copy the dataset from <href> http://47.93.21.181/lucaone/PreTrainingDataset/dataset/lucagplm </href> into the directory: `./dataset/` 
 
The training dataset(`dataset/lucagplm/v2.0/train/`) whose file names start with **'2023112418163521'** are gene data(DNA + RNA), and those that start with **'2023112314061479'** are protein data.

The validation dataset(`dataset/lucagplm/v2.0/dev/`) whose file names start with **'2023112418224620'** are gene data(DNA + RNA), and those that start with **'2023112314080544'** are protein data.

The testing dataset(`dataset/lucagplm/v2.0/test/`) whose file names start with **'2023112418231445'** are gene data(DNA + RNA), and those that start with **'2023112314083364'** are protein data.

**Notice**     
If you want to train individual nucleic acid or protein LucaOne(LucaOne-Gene or LucaOne-Prot), please separate the datasets as described above.   

## 9. Training Scripts   
Training scripts are under the directory `src/training`, including 4 shell scripts:    
`run_multi_v2.0.sh`:  nucleic acid(DNA+RNA) and protein mixed training with 10 pre-training tasks.   
`run_multi_mask_v2.0.sh`:  nucleic acid(DNA+RNA) and protein mixed training with only 2 mask pre-training tasks.       
`run_multi_v2.0_gene.sh`:  individual nucleic acid training with 3 pre-training tasks.   
`run_multi_v2.0_prot.sh`:  individual protein training with 7 pre-training tasks.      

### TensorBoard for Loss Curve     
tensorboard --logdir tb-logs --bind_all --port 8008

## 10. Continue Training when Failure     
`run_multi_v2.0_continue.sh`:  continue training when failure (i.e. an interruption occurs).   

## 11. Data and Code Availability     

**The Pretraining Dataset uploaded CNSA**      
The pre-training dataset of LucaOne has been deposited into CNGB Sequence Archive (CNSA) with accession number CNP0007266 (https://db.cngb.org/search/project/CNP0007266/).

**FTP:**   
Pre-training data, code, and trained checkpoint of LucaOne, embedding inference code, downstream validation tasks data & code, and other materials are available: <a href='http://47.93.21.181/lucaone/'>FTP</a>. 

**Details:**     

The LucaOne's model code is available at: <a href='https://github.com/LucaOne/LucaOne'>LucaOne Github </a> or <a href='http://47.93.21.181/lucaone/LucaOne/'>LucaOne</a>.   

The trained-checkpoint files are available at: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/latest/'>TrainedCheckPoint</a>.

LucaOne's representational inference code is available at: <a href='https://github.com/LucaOne/LucaOneApp'>LucaOneApp Github</a> or <a href='http://47.93.21.181/lucaone/LucaOneApp'>LucaOneApp</a>. 

The project of 8 downstream tasks is available at: <a href='https://github.com/LucaOne/LucaOneTasks'>LucaOneTasks Github</a> or <a href='http://47.93.21.181/lucaone/LucaOneTasks'>LucaOneTasks</a>.

The pre-training dataset of LucaOne is opened at: <a href='http://47.93.21.181/lucaone/PreTrainingDataset/'>PreTrainingDataset</a>. 

The datasets of downstream tasks are available at: <a href='http://47.93.21.181/lucaone/DownstreamTasksDataset/'> DownstreamTasksDataset </a>. 

The trained models of downstream tasks are available at: <a href='http://47.93.21.181/lucaone/DownstreamTasksTrainedModels/'> DownstreamTasksTrainedModels </a>.

Other supplementary materials are available at: <a href='http://47.93.21.181/lucaone/Others/'> Others </a>.

## 12. Contributor        
<a href="https://scholar.google.com.hk/citations?user=RDbqGTcAAAAJ&hl=en" title="Yong He">Yong He</a>, 
<a href="https://scholar.google.com/citations?user=lT3nelQAAAAJ&hl=en" title="Zhaorong Li">Zhaorong Li</a>, 
<a href="https://scholar.google.com/citations?view_op=list_works&hl=en&user=uvrzUfEAAAAJ" title="Yongtao Shan">Yongtao Shan</a>, 
<a href="https://scholar.google.com/citations?user=ODcOX4AAAAAJ&hl=zh-CN" title="Pan Fang">Pan Fang</a>, Yanhong Wei, 
<a href="https://scholar.google.com.hk/citations?hl=zh-CN&pli=1&user=Zhlg9QkAAAAJ" title="Yuan-Fei Pan">Yuan-Fei Pan</a>, 
<a href="https://scholar.google.com/citations?user=1KJOH7YAAAAJ&hl=zh-CN&oi=ao" title="Mang Shi">Mang Shi</a>, 
<a href="https://scholar.google.com/citations?hl=en&view_op=list_works&gmla=AGd7smFoGC01LG3CopJC_1HRW1Wpbk7W42IfwjCeac8GN2enJ8TEJ6t3JN5PVaugdD34CvNw3LJdUoWlY1XOpQ&user=Bd_HtNAAAAAJ" title="Jiaying Yang">Jiaying Yang</a>, 
<a href="https://scholar.google.com/citations?user=t0YyeMcAAAAJ&hl=zh-CN" title="Yihao Chen">Yihao Chen</a>,
<a href="https://scholar.google.com/citations?hl=en&user=ZmtOCdgAAAAJ" title="Yan Sun">Yan Sun</a>, Yuqi Liu  


## 13. Zenodo         
We have uploaded the model code, training scripts, and embedding inference scripts of LucaOne;    
The mode code, training and evaluation scripts, datasets, and trained models for downstream tasks,    
and additional supplementary materials to Zenodo (10.5281/zenodo.15171943).    
However, due to the substantial size of the pretraining dataset of LucaOne, it has not been included on Zenodo.     
Instead, it remains accessible via our publicly available FTP server (**<a href='http://47.93.21.181/lucaone/PreTrainingDataset/'>LucaOne Pretraining Dataset</a>**).     
We are actively seeking an open FTP platform with sufficient storage capacity to host our pretraining dataset.

**<a href='https://doi.org/10.5281/zenodo.15171943'>LucaOne Zenodo</a>**   


## 14. Citation         
**<a href='https://www.biorxiv.org/content/10.1101/2024.05.10.592927v2'>LucaOne Biorxiv</a>**   
**<a href='https://www.nature.com/articles/s42256-025-01044-4'>LucaOne NMI 2025</a>**


He, Y., Fang, P., Shan, Y. et al. Generalized biological foundation model with unified nucleic acid and protein language. Nat Mach Intell 7, 942–953 (2025). https://doi.org/10.1038/s42256-025-01044-4


## 15. LucaTeam

<center>
<img alt="LucaTeam" src="./pics/LucaTeam.jpg"/>

Fig. 5 LucaTeam at the West Lake in Hangzhou.
</center>   







