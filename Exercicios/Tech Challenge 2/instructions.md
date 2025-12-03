# conda create -n tf-gpu-new python=3.11

# conda env list

# conda activate tf-gpu-new

# conda activate tf-gpu-210

# conda remove tensorflow tensorflow-gpu --force

# conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1

# conda list | findstr "cudatoolkit cudnn"

# pip install tensorflow-gpu==2.10.0

# pip install numpy==1.23.5

# Para instalar pytorch
# pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html --no-deps

# pytorch é do facebook, mais simples e imperativo, mais simples e fácil, e tem python like code
# Tensor flow é da google, mais estruturado e voltado para produção, curva de aprendizado maior

# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
