#!/bin/sh
#env=StatelessAnt
#env=NoisyStatelessAntEasy
env=NoisyStatelessAntHard
#env=NoisyStatelessAntMedium
seed_max=20
for seed in `seq ${seed_max}`; do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../GPO.py --env_name ${env} --seed ${seed} \
    --unroll_length 5 --batch_size 1024 --num_minibatches 32 --num_update_epochs 4\
    --reward_scaling 0.1 --entropy_cost 0.01 --discounting 0.97 --target_kl 0.001 \
    --eps 0.3 --alpha 2 --use_clip True
done


