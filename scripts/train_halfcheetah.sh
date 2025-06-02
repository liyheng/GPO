#!/bin/sh
env=StatelessHalfcheetah
#env=NoisyStatelessHalfcheetahEasy
#env=NoisyStatelessHalfcheetahMedium
#env=NoisyStatelessHalfcheetahHard
seed_max=10
for seed in `seq ${seed_max}`; do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python GPO.py --env_name ${env} --seed ${seed} \
    --unroll_length 5 --batch_size 512 --num_minibatches 4 --num_update_epochs 4 \
    --reward_scaling 1 --entropy_cost 0.001 --discounting 0.99 --target_kl 0.001 \
    --eps 0.1 --alpha 2 --use_clip True
done

