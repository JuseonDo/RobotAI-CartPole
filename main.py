import os, time
import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from environment import DoublePoleCartEnv



def train(env_id='custom_cartpole', timesteps = 1000000, reward_type='custom_reward1', model_base_dir="models", model_name=None):

    seed = 256
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed)
    
    # Environment
    env = DoublePoleCartEnv(reward_type=reward_type)
    # env = make_vec_env(env_id, n_envs=2, seed=seed)
    check_env(env)

    # Agent Model
    if model_name is None:    
        model_name = env_id + "_PPO"
    model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./PPO_cartpole_tensorboard/")


    # Train
    # timesteps = 18000
    # timesteps = 1000000
    model.learn(
        total_timesteps=timesteps,
        callback=None,
        log_interval=4,
        tb_log_name="PPO",
        reset_num_timesteps=True,
        progress_bar=True,
    )

    # Save the trained model
    save_path = os.path.join(os.getcwd(), model_base_dir, model_name)
    model.save(save_path)

    # clase the environment
    env.close()


def run(env_id, reward_type='custom_reward1', model_base_dir="models", model_name=None):
    # Environment
    env = DoublePoleCartEnv(reward_type=reward_type)
    
    # Model
    if model_name is None:
        model_name = env_id + "_PPO"
    model_path = os.path.join(model_base_dir, model_name)
    model = PPO.load(model_path, env)

    # Run
    obs = env.reset()
    n_steps = 0     
    for i in range(1000):
        action, next_state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        time.sleep(0.01)
        env.render()
        n_steps += 1
        if done:
            time.sleep(1.0)
            print("n_steps: {}".format(n_steps))
            n_steps = 0
            obs = env.reset()
        
    env.close()


def exe(reward_type, timesteps = 1000000):
    env_id = "custom_cartpole"
    train(env_id,timesteps,reward_type)
    run(env_id,reward_type)

if __name__ == '__main__':
    # exe('default')
    # exe('custom_reward1')
    # exe(reward_type = 'custom_reward2')
    # exe(reward_type = 'custom_reward3')
    # exe(reward_type = 'custom_reward4')
    # exe(reward_type = 'custom_reward5')
    exe(reward_type = 'custom_reward6')
    exe(reward_type = 'custom_reward7')