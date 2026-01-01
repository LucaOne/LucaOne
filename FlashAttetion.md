# flash attention
conda create -n lucaone_flash_attention python=3.9.13
conda activate lucaone_flash_attention

pip install torch==2.5.1 torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu121
pip install psutil -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash-attn --no-build-isolation  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirement_others.txt -i https://pypi.tuna.tsinghua.edu.cn/simple