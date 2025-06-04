#!/bin/sh
env=StatelessWalker2d
#env=NoisyStatelessWalker2dEasy
#env=NoisyStatelessWalker2dMedium
#env=NoisyStatelessWalker2dHard
seed_max=20
for seed in `seq ${seed_max}`; do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../GPO.py --env_name ${env} --seed ${seed} --stack_frames 1 \
    --unroll_length 5 --batch_size 512 --num_minibatches 32 --num_update_epochs 4\
    --reward_scaling 1 --entropy_cost 0.01 --discounting 0.99 --target_kl 0.001 \
    --eps 0.1 --alpha 2 --use_clip True
done

