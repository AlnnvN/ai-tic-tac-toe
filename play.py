from tictactoe import TicTacToe
from stable_baselines3 import PPO
import numpy as np

def get_observation(board, chosen_side, opponent_side) -> dict:
        raw_board: list[list[str]] = board
        board: list[list[int]] = [[' ' for _ in range(3)] for _ in range(3)]

        for i in range(3):
            for j in range(3):
                access = raw_board[i][j]
                if access == ' ':
                    board[i][j] = 0
                elif access == chosen_side:
                    board[i][j] = 1
                elif access == opponent_side:
                    board[i][j] = 2

        observation ={
            'board': np.array(board, dtype=np.int32) 
        } 
        # print(f'observation: {observation}')
        return observation

def opponent_play(tictactoe: TicTacToe, model: PPO, raw_board, opponent_side, player_side) -> None:
    board = get_observation(raw_board, opponent_side, player_side)

    available = tictactoe.get_available_positions()
    while True:
        action, _state = model.predict(board)
        is_busy = False
        for pos in available:
            if pos[0] == action[0] and pos[1] == action[1]:
                is_busy = True
        if not is_busy:
            break
    
    tictactoe.play_round(action[0], action[1])

def main() -> None:
    my_side = 'O'

    opponent_side = ''

    if my_side == 'O':
        opponent_side = 'X'
    else:
        opponent_side = 'O'

    tictactoe = TicTacToe(my_side)

    model = PPO.load("data/rubiks_cube_data/best_model_on_timeset_1040000_on_mean_8.44")

    while tictactoe.check_result() == 'ongoing':
        tictactoe.print()
        row = int(input('row: '))
        column = int(input('column: '))
        tictactoe.play_round(row, column)
        opponent_play(tictactoe, model, tictactoe.board, opponent_side, my_side)
        print(f'status -> {tictactoe.check_result()}')


if __name__ == "__main__":
    main()