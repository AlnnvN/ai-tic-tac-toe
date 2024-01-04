import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tictactoe import TicTacToe
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
import random

class TicTacToeEnv(gym.Env):
    
    def __init__(self) -> None:
        self.chosen_side = 'X'

        self.opponent_side = ''

        if self.chosen_side == 'X':
            self.opponent_side = 'O'
        else:
            self.opponent_side = 'X'

        self.tictactoe: TicTacToe = TicTacToe(self.chosen_side)

        self.action_space: spaces.MultiDiscrete = self.__create_action_space()

        self.observation_space: spaces.Dict = self.__create_observation_space()

        self.step_count: int = 0


    def __create_action_space(self) -> spaces.MultiDiscrete:
        return spaces.MultiDiscrete([
                                3, #row
                                3, #column
                            ])

    def __create_observation_space(self) -> spaces.Dict:

        blank: int = 0
        me: int = 1
        opponent: int = 2

        min_value: int = blank
        max_value: int = opponent
        matrix_shape: tuple = (3, 3) 

        #3x3 board
        return spaces.Dict({
                            "board": spaces.Box(low=min_value, high=max_value, shape=matrix_shape, dtype=np.int32)
                         })
    
    def __get_observation(self) -> dict:
        raw_board: list[list[str]] = self.tictactoe.board
        board: list[list[int]] = [[' ' for _ in range(3)] for _ in range(3)]

        for i in range(3):
            for j in range(3):
                access = raw_board[i][j]
                if access == ' ':
                    board[i][j] = 0
                elif access == self.chosen_side:
                    board[i][j] = 1
                elif access == self.opponent_side:
                    board[i][j] = 2

        observation ={
            'board': np.array(board, dtype=np.int32) 
        } 
        # print(f'observation: {observation}')
        return observation

    def __execute_action(self, action: tuple) -> bool:
        row: int = action[0]
        column: int = action[1]

        return self.tictactoe.play_round(row, column)
    
    def __play_opponent(self) -> None:
        
        random.seed()

        available = self.tictactoe.get_available_positions()

        move = available[random.randrange(0, len(available))]
        # print(f'opponent action: {move}')

        self.tictactoe.play_round(move[0], move[1])
    
    def step(self, action: tuple):

        # print(f'my action: {action}')
        was_valid: bool = self.__execute_action(action)

        status: bool = self.tictactoe.check_result() # check if agent action ended game

        if status == 'ongoing' and was_valid:
            self.__play_opponent()
        
        self.state = self.__get_observation()

        status: bool = self.tictactoe.check_result() # check if opponent action ended game
        # print(f"STATUSSSS -> {status}")
        self.ep_reward = 0.0
        truncated = False
        info = {}

        if status == 'draw':
            self.ep_reward = 2
            self.terminated = True
        elif status == self.opponent_side:
            self.ep_reward = -10
            self.terminated = True
        elif status == self.chosen_side:
            self.ep_reward = 10
            self.terminated = True

        if not was_valid:
            self.ep_reward = -50
            self.terminated = True

        # self.tictactoe.print()
        # print(f"REWARD -> {self.ep_reward}\n\n\n")
            
        self.step_count += 1
    
        return self.state, self.ep_reward, self.terminated, truncated, info
    
    def reset(self, seed=0, options={}): 

        if self.step_count % 2000 == 0:
            self.tictactoe.print()
     
        self.tictactoe = TicTacToe(self.chosen_side)

        self.ep_reward = 0.0

        self.terminated = False

        self.state = self.__get_observation()
        
        info = {}

        return self.state, info



class TicTacToeCallBack(BaseCallback):
    def __init__(self, check_freq: int, saving_interval: int, log_dir: str, reward_threshold: float = 5.5, kill_on_timesteps: bool = False, max_timesteps: int = 0 ,verbose: int = 1):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.kill_on_timesteps = kill_on_timesteps
        self.max_timesteps = max_timesteps
        self.saving_timestep_interval = saving_interval
    
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self):
        forceSave = 0
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(f"{self.save_path}_on_mean_{mean_reward}")
                
        if self.num_timesteps % self.saving_timestep_interval == 0:
            # Example for saving best model
            if self.verbose >= 1:
                print(f"Saving new best model to {self.save_path}")
            self.model.save(f"{self.save_path}_on_timeset_{self.num_timesteps}_on_mean_{mean_reward}")
        
        with open("training_status.txt", "r") as f:
            if(f.read() == "2"):
                forceSave = 1
                x, y = ts2xy(load_results(self.log_dir), "timesteps")
                if len(x) > 0:
                    mean_reward = np.mean(y[-100:])
                    print(f"Forcing Model Save on {self.save_path}")
                    self.model.save(f"{self.save_path}_FORCED_on_mean_{mean_reward}")
        
        if(forceSave == 1):
            with open("training_status.txt", "w") as f:
                f.write("1")
                forceSave = 0
            
        if self.kill_on_timesteps == True:
            continue_training = False if self.n_calls >= self.max_timesteps else True
        else:
            continue_training = True if self.best_mean_reward < self.reward_threshold else False
        
        if not continue_training:
            self.model.save(f"{self.save_path}_overall")
            if self.verbose >= 1:
                print(
                    f"Stopping training because the mean reward {self.best_mean_reward:.2f} "
                    f" is above the threshold {self.reward_threshold}"
                )
        
        return continue_training