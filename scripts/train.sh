SIZE="T"

python train.py --cfg config/train/Tartan480x640-"$SIZE".json --savedir checkpoints/
python train.py --cfg config/train/Tartan-C368x496-"$SIZE".json --savedir checkpoints/ --restore_ckpt checkpoints/flowseek_"$SIZE"_Tartan.pth
python train.py --cfg config/train/Tartan-C-T432x960-"$SIZE".json --savedir checkpoints/ --restore_ckpt checkpoints/flowseek_"$SIZE"_TartanC.pth
python train.py --cfg config/train/Tartan-C-T-TSKH432x960-"$SIZE".json --savedir checkpoints/ --restore_ckpt checkpoints/flowseek_"$SIZE"_TartanCT.pth
