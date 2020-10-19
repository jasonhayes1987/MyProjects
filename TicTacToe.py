"""
Tic Tac Toe
Programmer: Jason Hayes
"""

gameBoard = [['/', '/', '/'], ['/', '/', '/'], ['/', '/', '/']]
game_win = False
turn = 1
player_x = []
player_o = []
player_selections = []


def new_game(board):
    i = 0
    for x, y in [board]:
        board[x][y] = '/'

    return board


# noinspection PyPep8
def build_board(board):
    print(board[0][0] + '|' + board[0][1] + '|' + board[0][2] + '| \n' + board[1][0] + '|' + board[1][1] + '|' + \
          board[1][2] + '| \n' + board[2][0] + '|' + board[2][1] + '|' + board[2][2] + '| \n')


def define_placement(board, num, piece):
    if num == '1':
        board[2][0] = piece
    elif num == '2':
        board[2][1] = piece
    elif num == '3':
        board[2][2] = piece
    elif num == '4':
        board[1][0] = piece
    elif num == '5':
        board[1][1] = piece
    elif num == '6':
        board[1][2] = piece
    elif num == '7':
        board[0][0] = piece
    elif num == '8':
        board[0][1] = piece
    elif num == '9':
        board[0][2] = piece
    return board

# noinspection PySimplifyBooleanCheck
def player_move(turn):
    # noinspection PySimplifyBooleanCheck
    num = input('Use the numpad to select the number that corresponds to the placement of your game piece')
    return num


def check_move(num):
    for selections in player_selections:
        if num == selections:
            print('there is already a piece there. Try again')
            return False
        elif num == '':
            print('you must select a valid number on the numpad')
            return False
    return True


def move_adder(piece, num):
    if piece == 'x':
        player_x.append(num)
        player_selections.append(num)
    else:
        player_o.append(num)
        player_selections.append(num)


def player_piece():
    if turn % 2 == 0:
        print(turn)
        return 'o'
    else:
        print(turn)
        return 'x'

def win_check(piece, board):
    global game_win
    if piece == 'x':
        if player_x.__contains__('7') and player_x.__contains__('8') and player_x.__contains__('9'):
            game_win = True
            print('you win')
        elif player_x.__contains__('4') and player_x.__contains__('5') and player_x.__contains__('6'):
            game_win = True
            print('you win')
        elif player_x.__contains__('1') and player_x.__contains__('2') and player_x.__contains__('3'):
            game_win = True
            print('you win')
        elif player_x.__contains__('7') and player_x.__contains__('4') and player_x.__contains__('1'):
            game_win = True
            print('you win')
        elif player_x.__contains__('8') and player_x.__contains__('5') and player_x.__contains__('2'):
            game_win = True
            print('you win')
        elif player_x.__contains__('9') and player_x.__contains__('6') and player_x.__contains__('3'):
            game_win = True
            print('you win')
        elif player_x.__contains__('7') and player_x.__contains__('5') and player_x.__contains__('3'):
            game_win = True
            print('you win')
        elif player_x.__contains__('9') and player_x.__contains__('5') and player_x.__contains__('1'):
            game_win = True
            print('you win')
    elif piece == 'o':
        if player_o.__contains__('7') and player_o.__contains__('8') and player_o.__contains__('9'):
            game_win = True
            print('you win')
        elif player_o.__contains__('4') and player_o.__contains__('5') and player_o.__contains__('6'):
            game_win = True
            print('you win')
        elif player_o.__contains__('1') and player_o.__contains__('2') and player_o.__contains__('3'):
            game_win = True
            print('you win')
        elif player_o.__contains__('7') and player_o.__contains__('4') and player_o.__contains__('1'):
            game_win = True
            print('you win')
        elif player_o.__contains__('8') and player_o.__contains__('5') and player_o.__contains__('2'):
            game_win = True
            print('you win')
        elif player_o.__contains__('9') and player_o.__contains__('6') and player_o.__contains__('3'):
            game_win = True
            print('you win')
        elif player_o.__contains__('7') and player_o.__contains__('5') and player_o.__contains__('3'):
            game_win = True
            print('you win')
        elif player_o.__contains__('9') and player_o.__contains__('5') and player_o.__contains__('1'):
            game_win = True
            print('you win')






def game():
    global turn
    global gameBoard
    while not game_win:
        piece = player_piece()
        num = player_move(turn)
        if check_move(num):
            move_adder(piece, num)
            define_placement(gameBoard,num,piece)
            turn += 1
        build_board(gameBoard)
        win_check(piece, gameBoard)





game()
