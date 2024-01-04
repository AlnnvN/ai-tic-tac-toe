import os, sys, time
from training_env import TicTacToeCallBack
from training_env import TicTacToeEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

def main():
    # Create log dir
    log_dir = "data/rubiks_cube_data"
    os.makedirs(log_dir, exist_ok=True)

    total_timesteps = int(2e7)
    reward_threshold = 8.5

    env = TicTacToeEnv()
    monitor = Monitor(env, log_dir)
    callback = TicTacToeCallBack(check_freq=2000, saving_interval=(total_timesteps*0.004), log_dir=log_dir, reward_threshold=reward_threshold * 1000, verbose=1)
    
    model = PPO("MultiInputPolicy", monitor, verbose=1, tensorboard_log="./tensorboard/", learning_rate=0.0003, gamma=0.995, batch_size=32)
    # model = PPO.load(f"{log_dir}/best_model/best_model_FORCED_on_mean_36.07")
    # model.set_env(monitor)
    model.learn(total_timesteps=total_timesteps, log_interval=2, progress_bar=True, callback=callback)

    print("Optimized model saved!")

    sys.exit(1)


    """
    **IF YOU WISH TO CONTINUE TRAINING FROM A SAVED NET DO THE FOLLOWING**

    model = PPO.load("saved_net")
    model.set_env(env)
    model.learn()

    """
    """
        **TESTING A TRAINED MODEL**

        # model = PPO.load("../data/kick_data_Backup/best_model_on_mean_144.83419916000003")
        # obs, info = env.reset()
        # while True: # Aqui dentro ele gera uma ação, manda essa ação pela função step e depois verifica se o dones voltou como true 
        #     action, _state = model.predict(obs)
        #     obs, rewards, terminated, truncated, info = env.step(action)
        #     print(f"Reward -> {rewards}")
        #     if(terminated):
        #         env.reset()

    """

if __name__ == "__main__":
    main()