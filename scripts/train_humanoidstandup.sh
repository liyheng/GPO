#!/bin/sh
#env=NoisyStatelessHumanoidStandupHard
#env=NoisyStatelessHumanoidStandupMedium
#env=NoisyStatelessHumanoidStandupEasy
env=StatelessHumanoidStandup
seed_max=20
for seed in `seq ${seed_max}`; do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../GPO.py --env_name ${env} --seed ${seed} \
    --unroll_length 5 --batch_size 256 --num_minibatches 32 --num_update_epochs 8\
    --reward_scaling 0.1 --entropy_cost 0.01 --discounting 0.99 --target_kl 0.001 \
    --eps 0.3 --alpha 3 --use_clip True
done
