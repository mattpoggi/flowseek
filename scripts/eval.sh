echo "TABLE 3 (CT)"
echo "Validation FlowSeek (T)"
python evaluate.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_CT.pth --dataset sintel
python evaluate.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_CT.pth --dataset kitti

echo "Validation FlowSeek (S)"
python evaluate.py --cfg config/eval/flowseek-S.json --model weights/flowseek_T_CT.pth --dataset sintel
python evaluate.py --cfg config/eval/flowseek-S.json --model weights/flowseek_T_CT.pth --dataset kitti

echo "Validation FlowSeek (M)"
python evaluate.py --cfg config/eval/flowseek-M.json --model weights/flowseek_M_CT.pth --dataset sintel
python evaluate.py --cfg config/eval/flowseek-M.json --model weights/flowseek_M_CT.pth --dataset kitti

echo "Validation FlowSeek (L)"
python evaluate.py --cfg config/eval/flowseek-L.json --model weights/flowseek_M_CT.pth --dataset sintel
python evaluate.py --cfg config/eval/flowseek-L.json --model weights/flowseek_M_CT.pth --dataset kitti

echo "TABLE 3 (Tartan+CT)"
echo "Validation FlowSeek (T)"
python evaluate.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT.pth --dataset sintel
python evaluate.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT.pth --dataset kitti

echo "Validation FlowSeek (S)"
python evaluate.py --cfg config/eval/flowseek-S.json --model weights/flowseek_T_TartanCT.pth --dataset sintel
python evaluate.py --cfg config/eval/flowseek-S.json --model weights/flowseek_T_TartanCT.pth --dataset kitti

echo "Validation FlowSeek (M)"
python evaluate.py --cfg config/eval/flowseek-M.json --model weights/flowseek_M_TartanCT.pth --dataset sintel
python evaluate.py --cfg config/eval/flowseek-M.json --model weights/flowseek_M_TartanCT.pth --dataset kitti

echo "Validation FlowSeek (L)"
python evaluate.py --cfg config/eval/flowseek-L.json --model weights/flowseek_M_TartanCT.pth --dataset sintel
python evaluate.py --cfg config/eval/flowseek-L.json --model weights/flowseek_M_TartanCT.pth --dataset kitti

echo "TABLE 4"
echo "Validation FlowSeek (T)"
python evaluate.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT_TSKH.pth --dataset spring --scale -1
echo "Validation FlowSeek (S)"
python evaluate.py --cfg config/eval/flowseek-S.json --model weights/flowseek_T_TartanCT_TSKH.pth --dataset spring --scale -1

echo "Validation FlowSeek (M)"
python evaluate.py --cfg config/eval/flowseek-M.json --model weights/flowseek_M_TartanCT_TSKH.pth --dataset spring --scale -1
echo "Validation FlowSeek (L)"
python evaluate.py --cfg config/eval/flowseek-L.json --model weights/flowseek_M_TartanCT_TSKH.pth --dataset spring --scale -1

echo "TABLE 5 + TABLE B (SUPPLEMENTARY)"
echo "Validation FlowSeek (T)"
python evaluate.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT_TSKH.pth --dataset layeredflow
echo "Validation FlowSeek (S)"
python evaluate.py --cfg config/eval/flowseek-S.json --model weights/flowseek_T_TartanCT_TSKH.pth --dataset layeredflow

echo "Validation FlowSeek (M)"
python evaluate.py --cfg config/eval/flowseek-M.json --model weights/flowseek_M_TartanCT_TSKH.pth --dataset layeredflow
echo "Validation FlowSeek (L)"
python evaluate.py --cfg config/eval/flowseek-L.json --model weights/flowseek_M_TartanCT_TSKH.pth --dataset layeredflow