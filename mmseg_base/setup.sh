cd ../
git submodule init
git submodule update
cd mmseg_base/mmsegmentation
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
pip install -v -e .
cd ../