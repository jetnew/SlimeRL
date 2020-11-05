#! /bin/sh
conda env create -f environment.yml
conda activate slime-rl
pip install ax-platform
pip install seaborn
pip install imageio-ffmpeg
pip install "stable-baselines[mpi]"
pip install slimevolleygym
pip install imageio