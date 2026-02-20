import numpy as np
import math
import random

BLACK = 1
WHITE = -1
BLANK = 0

def check_valid_location(x, y):
    return not (x < 0 or y < 0 or x > 7 or y > 7)

def get_valid_action(board, x, y, direction, color, opposite_color, blank=False):
    x_tmp = x + direction[0]
    y_tmp = y + direction[1]
    while(check_valid_location(x_tmp, y_tmp)):
        if board[x_tmp][y_tmp] == color:
            x_tmp += direction[0]
            y_tmp += direction[1]
        else:
            break
    else:
        return None
    if blank:
        if board[x_tmp][y_tmp] == opposite_color:
            return x, y
    else:
        if board[x_tmp][y_tmp] == BLANK:
            return x_tmp, y_tmp
    return None


def valid_actions_color(board, turn_color, opposite_turn_color):
    valid_actions = {}
    for i in range (8):
        for j in range (8):
            if board[i][j] == turn_color:
                for d_x in range(-1, 2):
                    for d_y in range(-1, 2):
                        if d_x or d_y:
                            if check_valid_location(i + d_x, j + d_y) and board[i + d_x][j + d_y] == opposite_turn_color:
                                valid_action = get_valid_action(board, i, j, (d_x, d_y), opposite_turn_color, turn_color)
                                if valid_action != None:
                                    if valid_action in valid_actions:
                                        valid_actions[valid_action].append((-1 * d_x, -1 * d_y))
                                    else:
                                        valid_actions[valid_action] = [(-1 * d_x, -1 * d_y)]
    return valid_actions

def valid_actions_opposite_color(board, turn_color, opposite_turn_color):
    valid_actions = {}
    for i in range (8):
        for j in range (8):
            if board[i][j] == opposite_turn_color:
                for d_x in range(-1, 2):
                    for d_y in range(-1, 2):
                        if d_x or d_y:
                            if check_valid_location(i + d_x, j + d_y) and board[i + d_x][j + d_y] == turn_color:
                                valid_action = get_valid_action(board, i, j, (-1 * d_x, -1 * d_y), opposite_turn_color, turn_color)
                                if valid_action != None:
                                    if valid_action in valid_actions:
                                        valid_actions[valid_action].append((d_x, d_y))
                                    else:
                                        valid_actions[valid_action] = [(d_x, d_y)]
    return valid_actions

def valid_actions_blank(board, turn_color, opposite_turn_color):
    valid_actions = {}
    for i in range (8):
        for j in range (8):
            if board[i][j] == BLANK:
                for d_x in range(-1, 2):
                    for d_y in range(-1, 2):
                        if d_x or d_y:
                            if check_valid_location(i + d_x, j + d_y) and board[i + d_x][j + d_y] == opposite_turn_color:
                                valid_action = get_valid_action(board, i, j, (d_x, d_y), opposite_turn_color, turn_color, blank=True)
                                if valid_action != None:
                                    if valid_action in valid_actions:
                                        valid_actions[valid_action].append((d_x, d_y))
                                    else:
                                        valid_actions[valid_action] = [(d_x, d_y)]
    return valid_actions

def set_color(board, x, y, directions, color, opposite_color):
    for direction in directions:
        x_tmp = x + direction[0]
        y_tmp = y + direction[1]
        while(board[x_tmp][y_tmp] == color):
            board[x_tmp][y_tmp] = opposite_color
            x_tmp += direction[0]
            y_tmp += direction[1]

counter_finish = 0
policy = {}
def max_value(board, a, b, depth): #black is max
    white_count = np.count_nonzero(board==WHITE)
    black_count = np.count_nonzero(board==BLACK)
    blank_count = 64 - white_count - black_count
    valid_actions = None
    if white_count < black_count:
        if white_count < blank_count:
            valid_actions = valid_actions_opposite_color(board, BLACK, WHITE)
        else:
            valid_actions = valid_actions_blank(board, BLACK, WHITE)
    else:
        if black_count < blank_count:
            valid_actions = valid_actions_color(board, BLACK, WHITE)
        else:
            valid_actions = valid_actions_blank(board, BLACK, WHITE)
    
    if depth == 0 or blank_count == 0:
        return black_count
    if len(valid_actions) == 0:
        counter_finish += 1
    if counter_finish == 2:
        counter_finish = 0
        return black_count
    if counter_finish == 1 and len(valid_actions) > 0:
        counter_finish = 0
    
    v = -math.inf
    good_action = None
    for valid_action in valid_actions.keys():
        new_board = board.copy()
        new_board[valid_action[0]][valid_action[1]] = BLACK
        set_color(new_board, valid_action[0], valid_action[1], valid_actions[valid_action], WHITE, BLACK)
        new_v = max(v, min_value(new_board, a, b, depth - 1))
        if new_v > v:
            good_action = [valid_action, valid_actions[valid_action]]
        v = max(v, new_v)
        if v >= b:
            break
        a = max(a, v)
    if good_action == None:
        new_board = board.copy()
        v = min_value(new_board, a, b, depth)
    board_tuple = tuple(map(tuple, board))
    if board_tuple not in policy:
        policy[board_tuple] = good_action
    return v

def min_value(board, a, b, depth): #white is min
    white_count = np.count_nonzero(board==WHITE)
    black_count = np.count_nonzero(board==BLACK)
    blank_count = 64 - white_count - black_count
    valid_actions = None
    if white_count < black_count:
        if white_count < blank_count:
            valid_actions = valid_actions_color(board, WHITE, BLACK)
        else:
            valid_actions = valid_actions_blank(board, WHITE, BLACK)
    else:
        if black_count < blank_count:
            valid_actions = valid_actions_opposite_color(board, WHITE, BLACK)
        else:
            valid_actions = valid_actions_blank(board, WHITE, BLACK)
    
    if depth == 0 or blank_count == 0:
        return black_count
    if len(valid_actions) == 0:
        counter_finish += 1
    if counter_finish == 2:
        counter_finish = 0
        return black_count
    if counter_finish == 1 and len(valid_actions) > 0:
        counter_finish = 0
    
    v = math.inf
    for valid_action in valid_actions.keys():
        new_board = board.copy()
        new_board[valid_action[0]][valid_action[1]] = WHITE
        set_color(new_board, valid_action[0], valid_action[1], valid_actions[valid_action], BLACK, WHITE)
        v = min(v, max_value(new_board, a, b, depth - 1))
        if v <= a:
            return v
        b = min(b, v)
    if len(valid_actions) == 0:
        v = max_value(board, a, b, depth)
    return v


def minimax_without_policy(board, depth, a, b, turn, opposite_turn, finish, root=False):
    turn_count = np.count_nonzero(board==turn)
    opposite_turn_count = np.count_nonzero(board==opposite_turn)
    blank_count = 64 - turn_count - opposite_turn_count
    valid_actions = None
    if turn_count < opposite_turn_count:
        if turn_count < blank_count:
            valid_actions = valid_actions_color(board, turn, opposite_turn)
        else:
            valid_actions = valid_actions_blank(board, turn, opposite_turn)
    else:
        if opposite_turn_count < blank_count:
            valid_actions = valid_actions_opposite_color(board, turn, opposite_turn)
        else:
            valid_actions = valid_actions_blank(board, turn, opposite_turn)
    
    black_count = np.count_nonzero(board==BLACK)
    if depth == 0 or blank_count == 0:
        return black_count
    if len(valid_actions) == 0:
        finish[0] += 1
    if finish[0] == 2:
        finish[0] = 0
        return black_count
    if finish[0] == 1 and len(valid_actions) > 0:
        finish[0] = 0

    if turn == WHITE:
        v = math.inf
        for valid_action in valid_actions.keys():
            new_board = board.copy()
            new_board[valid_action[0]][valid_action[1]] = WHITE
            set_color(new_board, valid_action[0], valid_action[1], valid_actions[valid_action], BLACK, WHITE)
            v = min(v, minimax_without_policy(new_board, depth - 1, a, b, BLACK, WHITE, finish))
            if v <= a:
                return v
            b = min(b, v)
        if len(valid_actions) == 0:
            new_board = board.copy()
            v = minimax_without_policy(new_board, depth, a, b, BLACK, WHITE, finish)
        return v
    else:
        v = -math.inf
        good_action = None
        for valid_action in valid_actions.keys():
            new_board = board.copy()
            new_board[valid_action[0]][valid_action[1]] = BLACK
            set_color(new_board, valid_action[0], valid_action[1], valid_actions[valid_action], WHITE, BLACK)
            new_v = max(v, minimax_without_policy(new_board, depth - 1, a, b, WHITE, BLACK, finish))
            if new_v > v:
                good_action = [valid_action, valid_actions[valid_action]]
            v = max(v, new_v)
            if v >= b:
                break
            a = max(a, v)
        if good_action == None:
            new_board = board.copy()
            v = minimax_without_policy(new_board, depth, a, b, WHITE, BLACK, finish)
        if root:
            return good_action
        else:
            return v
        
def move_random(board): # for black or agent
    white_count = np.count_nonzero(board==WHITE)
    black_count = np.count_nonzero(board==BLACK)
    blank_count = 64 - white_count - black_count
    valid_actions = None
    if white_count < black_count:
        if white_count < blank_count:
            valid_actions = valid_actions_opposite_color(board, BLACK, WHITE)
        else:
            valid_actions = valid_actions_blank(board, BLACK, WHITE)
    else:
        if black_count < blank_count:
            valid_actions = valid_actions_color(board, BLACK, WHITE)
        else:
            valid_actions = valid_actions_blank(board, BLACK, WHITE)
    if len(valid_actions) == 0:
        return None
    else:
        random_action = random.choice(list(valid_actions.keys()))
        return [random_action, valid_actions[random_action]]
    
def winner(board):
    white_count = np.count_nonzero(board==WHITE)
    black_count = np.count_nonzero(board==BLACK)
    if black_count > white_count:
        return BLACK
    else:
        if black_count < white_count:
            return WHITE
        else:
            return BLANK
        
def display(board):
    print("   ", end=' ')
    for i in range(8):
        print(f"{i+1} ", end=' ')
    print('\n', end=' ')
    for i in range(7, -1, -1):
        print(f"{i + 1} ", end=' ')
        for j in range(8):
            if board[i][j] == BLACK:
                print("B ", end=' ')
            else:
                if board[i][j] == WHITE:
                    print("W ", end=' ')
                else:
                    print(". ", end=' ')
        print('\n', end=' ')
    
def display_blank(board, valid_actions):
    print("   ", end=' ')
    for i in range(8):
        print(f"{i+1} ", end=' ')
    print('\n', end=' ')
    for i in range(7, -1, -1):
        print(f"{i + 1} ", end=' ')
        for j in range(8):
            if board[i][j] == BLACK:
                print("B ", end=' ')
            else:
                if board[i][j] == WHITE:
                    print("W ", end=' ')
                else:
                    if (i,j) in valid_actions:
                        print("* ", end=' ')
                    else:
                        print(". ", end=' ')
        print('\n', end=' ')

board=np.zeros((8,8))

board[3][3]=BLACK
board[3][4]=WHITE
board[4][3]=WHITE
board[4][4]=BLACK

turn = BLACK
number = 5
finish_step = 0
while(True):
    if finish_step == 2 or number == 65:
        display(board)
        win = winner(board)
        if win == BLACK:
            print("agent wins.")
        else:
            if win == WHITE:
                print("You wins.")
            else:
                print("equality!!")
        break
    if turn == BLACK:
        display(board)
        #action = move_random(board)
        if number > 54:
            action = minimax_without_policy(board, 10, -math.inf, math.inf, BLACK, WHITE, [0], True)
        else:
            if number < 10:
                action = minimax_without_policy(board, 7, -math.inf, math.inf, BLACK, WHITE, [0], True)
            else:
                action = minimax_without_policy(board, 5, -math.inf, math.inf, BLACK, WHITE, [0], True)
        if action == None:
            print("There isn't valid action for agent!")
            finish_step += 1
        else:
            number += 1
            if finish_step == 1:
                finish_step = 0
            x = action[0][0]
            y = action[0][1]
            directions = action[1]
            set_color(board, x, y, directions, WHITE, BLACK)
            board[x][y] = BLACK
            print(f"agent action: ({x + 1}, {y + 1})")
        turn = WHITE
    else:
        valid_actions = valid_actions_color(board, WHITE, BLACK)
        display_blank(board, valid_actions)
        if len(valid_actions) == 0:
            print("There isn't valid action for you!")
            finish_step += 1
        else:
            inp = None
            while(True):
                inp = input("Enter your action: ")
                try:
                    inp = inp.split(',')
                    inp[0] = int(inp[0])
                    inp[1] = int(inp[1])
                except:
                    print("Please enter correct format x, y")
                    continue
                if len(inp) != 2:
                    print("Please enter correct format x, y")
                    continue
                inp[0] -= 1
                inp[1] -= 1
                inp = tuple(inp)
                if inp in valid_actions:
                    break
                else:
                    print("this action isn't correct!enter again:")
                    continue
            if finish_step == 1:
                finish_step = 0
            set_color(board, inp[0], inp[1], valid_actions[inp], BLACK, WHITE)
            board[inp[0]][inp[1]] = WHITE
            number += 1
        turn = BLACK

        

