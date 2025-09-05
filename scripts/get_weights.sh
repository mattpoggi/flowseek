mkdir -p weights/
wget 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth' -O weights/depth_anything_v2_vits.pth
wget 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth' -O weights/depth_anything_v2_vitb.pth

gdown --fuzzy 'https://drive.google.com/file/d/1COOQFkMulzpBm4zMoWsaRGk7E3YcVr2I/view?usp=share_link' -O weights/flowseek_T_CT.pth
gdown --fuzzy 'https://drive.google.com/file/d/1fEso4npxe1YBIAYPYZ8B6DM9hTO_Rvar/view?usp=share_link' -O weights/flowseek_M_CT.pth
gdown --fuzzy 'https://drive.google.com/file/d/1dnMzlRqX7wziynQvZjafsXjgbfibNAKG/view?usp=share_link' -O weights/flowseek_T_TartanCT.pth
gdown --fuzzy 'https://drive.google.com/file/d/1L8PDDkPJguu6qMSrfB7L7FdtZ0A5zFeQ/view?usp=share_link' -O weights/flowseek_M_TartanCT.pth
gdown --fuzzy 'https://drive.google.com/file/d/1IQoyY5PpKSadtiGuhWwVCqvgD3y8CyFd/view?usp=share_link' -O weights/flowseek_T_TartanCT_TSKH.pth
gdown --fuzzy 'https://drive.google.com/file/d/1gbZ-6NE3muAnGqvypiS2s_BADHrI4ySf/view?usp=share_link' -O weights/flowseek_M_TartanCT_TSKH.pth