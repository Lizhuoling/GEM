CUDA_VERSION=12.5

conda create -n sonata python=3.8
conda activate sonata

pip install torch==2.3.1 torchvision==0.18.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install spconv-cu120
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
#pip install git+https://github.com/Dao-AILab/flash-attention.git
pip install huggingface_hub timm
pip install einops


# (optional, or directly copy the sonata folder to your project)
python setup.py install

pip install open3d fast_pytorch_kmeans psutil numpy==1.23.5  # currently, open3d does not support numpy 2.x