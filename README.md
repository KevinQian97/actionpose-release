# actionpose-release
This is the release repo of our course project for 16824 2021 Spring

Authors: Yijun Qian, Shengcao Cao
Email: yijunqia@andrew.cmu.edu, shengcao@andrew.cmu.edu

## Pretrained Models:
We provided the pretrained models we reported for this project:[link](https://drive.google.com/drive/folders/1KNgQtQxauvEvQDCB3zCXFp001bj5NVSK?usp=sharing)

## Logs
You can find the log files and tensorboard events under ./logs folder

## Error Samples
You can find the error samples under ./error_samples folder

## Traning Script
```
python main.py --dataset NTU --modality PoseAction \
    --arch R2plus1D --is_3D --num_segments 32 \
    --gd 20 --lr 0.001 --lr_steps 4 6 8 10 --epochs 12 \
    --batch-size 20 -j 20 --dropout 0.5 --eval-freq=1 \
    --loss_type nll --gpus 0 1 2 3 --optimizer sgd --lr_scheduler \
    --root_model /home/kevinq/models/NTU \
    --pretrain r2plus1d_34_32_kinetics
```
## Testing Script
```
python main.py --dataset NTU --modality RGB \
    --arch R2plus1D --is_3D --num_segments 32 \
    --gd 20 --lr 0.001 --lr_steps 4 6 8 10 --epochs 12 \
    --batch-size 20 -j 20 --dropout 0.5 --eval-freq=1 \
    --loss_type nll --gpus 0 1 2 3 --optimizer sgd --lr_scheduler \
    --root_model /home/kevinq/models/NTU \
    --pretrain r2plus1d_34_32_kinetics \
    --resume /home/kevinq/models/NTU/TSA_NTU_RGB_R2plus1D_avg_segment32_e12/9.pth.tar \
    --evaluate  
```
