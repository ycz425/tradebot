from trade_env import TradeEnv
from stable_baselines3 import DQN
import torch
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import matplotlib.pyplot as plt
import numpy as np
import os

TIMESTEPS_PER_EPISODE = 1260
SAVE_DIR = 'dqn/models'


def train(model: DQN, save_name: str, timesteps: int, eval_env: VecMonitor, eval_freq: int) -> None:
    save_dir = os.path.join(SAVE_DIR, save_name)
    os.makedirs(save_dir, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=eval_freq,
        log_path=save_dir,
        best_model_save_path=save_dir,
    )
    model.learn(total_timesteps=timesteps, progress_bar=True, callback=eval_callback)


def load_evaluations(path: str) -> None:
    evals = np.load(path)
    timesteps = evals['timesteps']
    ep_rew_means = np.mean(evals['results'], axis=1)
    best_ep_rew_mean = np.max(ep_rew_means)
    best_ep_rew_mean_timestep = timesteps[np.argmax(ep_rew_means)]

    print(f'Best episode reward mean: {best_ep_rew_mean}, timestep: {best_ep_rew_mean_timestep}')

    plt.plot(timesteps, ep_rew_means)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Mean Reward")
    plt.show()


if __name__ == '__main__':    
    build_train_env = lambda: TradeEnv('MSFT', '2019-01-01', '2024-03-01', 0.03, 'dqn/data/MSFT.US_2019-01-01_to_2025-01-20.json')
    build_eval_env = lambda: TradeEnv('MSFT', '2019-01-01', '2024-03-01', 0.03, 'dqn/data/MSFT.US_2019-01-01_to_2025-01-20.json', eval=True)

    train_env = VecMonitor(SubprocVecEnv([build_train_env for _ in range(4)]))
    eval_env = VecMonitor(SubprocVecEnv([build_eval_env for _ in range(4)]))

    model = DQN(
        'MlpPolicy',
        train_env,
        learning_rate=0.0005,
        buffer_size=1000000,
        exploration_fraction=0.5,
        target_update_interval=1000,
        max_grad_norm=0.5,
        verbose=1,
        policy_kwargs=dict(
            net_arch=[64, 64, 64],
            activation_fn=torch.nn.ReLU
        )
    )

    train(model, 'tradebot_v1', TIMESTEPS_PER_EPISODE * 2000, eval_env, TIMESTEPS_PER_EPISODE * 4)
    load_evaluations('dqn/models/tradebot_v1/evaluations.npz')

    
    