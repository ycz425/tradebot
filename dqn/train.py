from trade_env import TradeEnv
from stable_baselines3 import DQN
import gymnasium as gym
import torch
from stable_baselines3.common.evaluation import evaluate_policy


def train(model: DQN, timesteps: int, val_env: gym.Env, val_interval: int, patience: int, save_path: str) -> None:
    best_mean = 0
    best_std = 0
    patience_count = 0
    for _ in range(timesteps):
        patience_count += 1
        model.learn(total_timesteps=val_interval, progress_bar=True)
        mean, std = evaluate_policy(model, val_env, n_eval_episodes=8)
        print(f'[Validation] mean: {mean}, std: {std}')
        if mean > best_mean:
            best_mean = mean
            patience_count = 0
        if std < best_std:
            best_std = std
            patience_count = 0
    
        if patience_count == patience:
            print('Early-stopping criterion met. Ending training.')
            break

    model.save(save_path)


if __name__ == '__main__':    
    train_env = TradeEnv('MSFT', '2019-01-01', '2024-03-01', 0.03, 'dqn/data/MSFT.US_2019-01-01_to_2025-01-20.json')
    val_env = TradeEnv('MSFT', '2024-01-01', '2025-01-20', 0.03, 'dqn/data/MSFT.US_2019-01-01_to_2025-01-20.json')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DQN(
        'MultiInputPolicy',
        train_env,
        verbose=1
    )

    train(model, 2000000, val_env, 10000, 20, 'dqn/models/tradebot')

    
    