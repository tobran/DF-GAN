# configs of different datasets
cfg=$1

# model settings
imsize=256
num_workers=4
batch_size_per_gpu=32
stamp=normal
train=True

# resume training
resume_epoch=1
resume_model_path=./saved_models/bird/base_z_dim100_bird_256_2022_06_04_23_20_33/

# DDP settings
multi_gpus=True
nodes=1
master_port=11111

# You can set CUDA_VISIBLE_DEVICES=0,1,2... to accelerate the training process if you have multiple GPUs
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$nodes --master_port $master_port src/train.py \
                    --stamp $stamp \
                    --cfg $cfg \
                    --batch_size $batch_size_per_gpu \
                    --num_workers $num_workers \
                    --imsize $imsize \
                    --resume_epoch $resume_epoch \
                    --resume_model_path $resume_model_path \
                    --train $train \
                    --multi_gpus $multi_gpus \

