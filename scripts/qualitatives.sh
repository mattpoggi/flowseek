echo "FIGURE 3"
python demo.py --cfg config/eval/flowseek-L.json --model weights/flowseek_M_TartanCT.pth --dataset sintel --id 880
python demo.py --cfg config/eval/flowseek-L.json --model weights/flowseek_M_TartanCT.pth --dataset sintel --id 75

echo "FIGURE 4"
python demo.py --cfg config/eval/flowseek-L.json --model weights/flowseek_M_TartanCT.pth --dataset kitti --id 22
python demo.py --cfg config/eval/flowseek-L.json --model weights/flowseek_M_TartanCT.pth --dataset kitti --id 49

echo "FIGURE C"
python demo.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT.pth --dataset sintel --id 200
python demo.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT.pth --dataset sintel --id 345
python demo.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT.pth --dataset sintel --id 440

echo "FIGURE D"
python demo.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT.pth --dataset kitti --id 10
python demo.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT.pth --dataset kitti --id 85
python demo.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT.pth --dataset kitti --id 115
python demo.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT.pth --dataset kitti --id 170

echo "FIGURE E"
python demo.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT_TSKH.pth --dataset spring --id 11920 --scale -1
python demo.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT_TSKH.pth --dataset spring --id 650 --scale -1

echo "FIGURE F"
python demo.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT.pth --dataset spring --id 7461 --scale -1
python demo.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT.pth --dataset spring --id 3000 --scale -1
python demo.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT.pth --dataset spring --id 1200 --scale -1
python demo.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT.pth --dataset spring --id 14350 --scale -1
