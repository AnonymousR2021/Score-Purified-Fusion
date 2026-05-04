# Score-Purified Fusion
Our work is based on GeminiFusion for Multimodal Segmentation on NYUDv2 and SUN RGB-D datasets (ICML 2024), and follows its overall framework and experimental setup.

We adopt the Swin-Large (window size = 12) backbone for feature extraction.

CUDA_VISIBLE_DEVICES=0,1,2 \
python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py \
--backbone swin_large_window12 \
--dataset nyudv2 \
-c nyudv2_swin_large_window12 \
--dpr 0.2
