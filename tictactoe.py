class TicTacToe:
    def __init__(self, first_to_play: str = 'O') -> None:
        self.current_to_play: str = first_to_play
        
        self.board: list[list[str]] = []

        self.__build_board()

        self.round_count: int = 0
        
    def __build_board(self) -> None:
        self.board = [
            [' ',' ',' '],
            [' ',' ',' '],
            [' ',' ',' ']
        ]        

    def print(self) -> None:
        count = 1
        print("  A  B  C ")
        for line in self.board:
            print(count, end='')
            for item in line:
                print("|" + item + "|", end='')
    
            print('')

            count+=1

    def play_round(self, row: int, column: int) -> bool:
        if row >= 3 or column >= 3:
            return False

        if self.board[row][column] != ' ':
            return False

        self.board[row][column] = self.current_to_play
        self.round_count += 1
        
        if self.current_to_play == 'O':
            self.current_to_play = 'X'
        else:
            self.current_to_play = 'O'

        return True

    def __check_win(self) -> bool:

        diagonal1: bool = False
        diagonal2: bool = False
        row_col: bool = False

        if self.board[1][1] != ' ':
            diagonal1 = (self.board[0][0] == self.board[1][1]) and (self.board[1][1] == self.board[2][2])
            diagonal2 = (self.board[2][0] == self.board[1][1]) and (self.board[1][1] == self.board[0][2])

        for i in range(3):
            condition_row = (self.board[i][0] == self.board[i][1] and self.board[i][1] == self.board[i][2])
            condition_column = (self.board[0][i] == self.board[1][i] and self.board[1][i] == self.board[2][i])
           
            if self.board[i][0] == ' ':
                condition_row = False
            if self.board[0][i] == ' ':
                condition_column = False

            if(condition_row or condition_column):
                row_col = True
                break

        win: bool = diagonal1 or diagonal2 or row_col

        return win

    def check_result(self) -> str:
        win: bool = self.__check_win()

        if win:
            #returns the winning player -> O or X
            #the winner is the last round's player
            if self.current_to_play == 'X':
                return 'O'
            else:
                return 'X'
        elif self.round_count == 9:
            #no win but board is complete -> draw
            return 'draw'
        else:
            #board not complete and no win -> ongoing
            return 'ongoing'

    def get_available_positions(self) -> 'list[tuple[int]]':
        available: list[tuple[int]] = []

        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    available.append((i, j))

        return available
