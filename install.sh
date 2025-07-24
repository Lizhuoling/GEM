conda create -n GEM python=3.8
conda activate GEM

# For simulation experiments, install Isaac Gym
# Move to isaacgym/python/
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# Move to the ROOT of this repo
pip install torch==2.3.1 torchvision==0.18.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cd utils/detr && pip install -e . && cd ../..

# Install sonata
cd utils/sonata
pip install spconv-cu120
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
pip install huggingface_hub==0.23.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup.py develop
cd ../..

python setup.py develop