import argparse

def get_config():
    parser = argparse.ArgumentParser(
        description='gpo', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--env_name", type=str, default='ant')
    parser.add_argument("--device", type=str, default='cuda', help="by default, will use GPU to train")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--num_envs", type=int, default=2048, help="Number of parallel envs for training rollouts")
    parser.add_argument("--episode_length", type=int, default=1000, help="Max length of an episode")
    parser.add_argument("--num_timesteps", type=int, default=30000000, help="Total environmental steps to train")
    parser.add_argument("--eval_frequency", type=int, default=10, help="The total number of evaluations")
    parser.add_argument("--unroll_length", type=int, default=5, help="The unroll length of the environment")
    parser.add_argument("--batch_size", type=int, default=512, help="Size of each batch")
    parser.add_argument("--num_minibatches", type=int, default=4, help="Number of batches")
    parser.add_argument("--num_update_epochs", type=int, default=4, help="Number of epochs for PPO")
    parser.add_argument("--reward_scaling", type=float, default=1., help="Reward scaling factor")
    parser.add_argument("--entropy_cost", type=float, default=1e-3, help="Entropy term coefficient")
    parser.add_argument("--discounting", type=float, default=0.99, help="Discount factor for rewards")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for both actor and critic")
    parser.add_argument("--target_kl", type=float, default=0.01, help="Desired KL divergence for GPO-penalty")
    parser.add_argument("--eps", type=float, default=0.2, help="Inner clip parameters for GPO-clip")
    parser.add_argument("--alpha", type=float, default=2., help="Coefficient of RL auxiliary loss GPO-clip")
    parser.add_argument("--use-clip", type=bool, default=True, help="by default, will use GPO-clip. Otherwise use GPO-penalty")

    return parser