mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

conda create -n -y neurips-llm python==3.10.0

conda activate neurips-llm

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118

cd lit-gpt
pip install -r requirements.txt
cd ..