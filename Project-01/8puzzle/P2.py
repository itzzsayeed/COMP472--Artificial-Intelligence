#!/usr/bin/env python3
import copy, math, bisect, time
from collections import deque

visited = []

mapping = [
    0,1,2,5,8,7,6,3,4 # for goal state 1,2,3,8,B,4,7,6,5
    # 0,1,2,3,4,5,6,7,8 # for goal state 1,2,3,4,5,6,7,8,B
]
GOAL = [
    1, 2, 3,
    8, 9, 4,
    7, 6, 5
]

tiles = 0
goal = 1
h_val = 2
h_func = 3
e_pos = 4
parent = 5
cost = 6


"""
1 2 3     1 2 3
4 5 6  -> 8 E 4
7 8 9     7 6 5
"""

def get_col(value):
    """
    Compute the column number from the value of the tile
    """
    return get_position(value) % 3

def get_row(value):
    """
    Compute the row number from the value of the tile
    """
    return get_position(value) // 3

def get_position(value):
    """
    Resolve the actual expected position from the value of the tile
    """
    return mapping[value - 1]


# Mapping to generate moves and verify if the move is legit
N = (-3, lambda X: X > 2)
S = (+3, lambda X: X < 6)
E = (+1, lambda X: (X % 3) < 2)
W = (-1, lambda X: (X % 3) > 0)


def inversions(state):
    """
    Compute the number of inversions in one state
    Find the number of misplaced tiles after each tile
    not counting the empty tile
    """
    count = 0
    for (i, cur) in enumerate(state[tiles]):
        # ignore blank tile
        if cur == 9: continue
        for x in state[tiles][i:]:
            if (get_position(x) < get_position(cur)
                and x != 9):
                count += 1
    return count

def is_solvable(state):
    """
    an 8-Puzzle is solvable if the number of inversions is even
    """
    return (inversions(state) % 2) == 0

def move(orig_state, D):
    """
    Move the empty tile in Direction D
    D: tuple (value, function)
        - value: to add/subtract to the empty position tile
        - function: to check move validity
    """
    empty = orig_state[e_pos]
    if D[1](empty):
        # deepcopy the state to avoid modifying a mutable
        state = copy.copy(orig_state)
        state[tiles] = orig_state[tiles][:]
        # Swap the target tile with empty
        state[tiles][empty] = \
            state[tiles][empty + D[0]]
        state[tiles][empty + D[0]] =  9
        # Update the empty tile position
        state[e_pos] = empty + D[0]
        state[cost] += 1
        # Compute heuristics
        heuristic(state)
        return state
    return None

def successors(state):
    """
    Generate all the possible successors of the state
    """
    s = []
    for m in [N, W, E, S]:
        res = move(state, m)
        if res is not None:
            s.append(res)
    return s

def is_goal(state):
    return state[tiles] == state[goal]

def heuristic(state):
    """
    Wrapper function that verifies the state validity or goal
    before calling the actual heuristic value
    """
    v = -1;
    if not is_solvable(state):
        v = math.inf
    elif is_goal(state):
        v = 0
    else:
        v =  state[h_func](state)
    state[h_val] = v

def hamming(state):
    """
    Number of misplaced tiles
    """
    return sum([
        1 for (i, v) in enumerate(state[tiles])
        if get_position(v) != i])


def manhattan(state):
    """
    heuristic adds the minimum number of moves for each tile
    """
    h = 0;
    for (current_pos, tile) in enumerate(state[tiles]):
        if tile == 9: continue
        expected_row = get_row(tile)
        expected_col = get_col(tile)
        actual_row = current_pos // 3
        actual_col = current_pos % 3
        h += (
            abs(expected_row - actual_row)
            + abs(expected_col - actual_col))
    return h

def permutation_inversion(state):
    """
    permutation heuristic is returning the same content as the number of inversions
    """
    return inversions(state)

def get_row_number(t, n):
    """
    Helper function to retrieve the row number *n* from a tile representation t
    """
    return t[n*3:(n+1)*3]

def get_col_number(t, n):
    """
    Helper function to retrieve the column number *n* from a tile representation t
    """
    return t[n:3:n+6]


def count_all_lc(state):
    """
    Compute all the linear conflicts in a state
    """
    count = 0
    for i in range(3):
        count += count_row_lc(
            get_row_number(state[tiles], i),
            i
        )
        count += count_col_lc(
            get_col_number(state[tiles], i),
            i
        )
    return count

def count_one_lc(triplet, cur_x, x_func, y_func):
    """
    Count linear conflict for one line (either row or column)
    triplet: tiles in the line
    cur_x:  current row or column number
    x_func: function that gives the expected position of a tile in the current row/col
    y_funct: like x_func but for the other dimension
    """
    in_target = []
    prev = None
    for (idx, val) in enumerate(triplet):
        if ( cur_x == x_func(val)
             and idx != y_func(val)):
            in_target.append(idx)
    if len(in_target) > 1:
        return len(in_target)
    return 0


def count_col_lc(triplet, col_num):
    """
    Count the linear conflicts in one column
    """
    return count_one_lc(
        triplet, col_num, get_col, get_row
    )

def count_row_lc(triplet, row_num):
    """
    Count the linear conflicts in one row
    """
    return count_one_lc(
        triplet, row_num, get_row, get_col
    )

def manhat_collision(state):
    """
    Heuristic that combines manhattan and the collision detection
    """
    return manhattan(state) + count_all_lc(state)


# FIFO
def bfs(state):
    """
    BFS: insert left, retrieve right == FIFO
    """
    return generic_search(state)

# Stack
def dfs(state):
    """
    DFS: insert left, retrieve left == Stack
    """
    return generic_search(state, lambda x: x.pop())

def get_solution(closed):
    """
    Generate the solution path, stating from bottom
    of closed to the initial state
    """
    sol = deque()
    child = closed[len(closed) - 1]
    while (child[parent] is not None):
        sol.appendleft(child)
        child = closed[child[parent]]
    # append initial state
    sol.appendleft(child)
    return sol

def generic_search(
        state,
        pop_func = lambda x: x.popleft(),
        sort_func = lambda x,y: x.append(y),
):
    """
    pop_func: function to remove next state from 'opened' list
    sort_func: function to insert successors into 'opened' list
    """
    opened = deque()
    closed = deque()
    time_in = time.perf_counter()
    current_state = state
    opened.append(state)
    found = False
    if not is_solvable(state): return (-1,-1,-1)
    while(not found and len(opened) > 0):

        current_state = pop_func(opened)
        # We found the goal state: stop the search
        if is_goal(current_state):
            found = True
            closed.append(current_state)
            break
        closed.append(current_state)
        for successor in successors(current_state):
            if (successor[h_val] != math.inf and
                    successor not in closed and
                successor not in opened):
                # Insert the successor sorted if desired
                successor[parent] = len(closed) - 1
                sort_func(
                    opened,
                    successor)
    time_out = time.perf_counter()
    return (
        len(closed),
        (time_out - time_in)*1000,
        len(get_solution(closed))
    )


def b1s(state):
    """
    Best First Search
    State: has to be initialized with a heuristic
    pop_funct: retrieves the state with smallest heuristic value first
    sort_func: inserts the successor in opened ordered by heuristic values
    """
    return generic_search(
        state,
        pop_func=lambda x: x.popleft(),
        sort_func=lambda _opened, _successor: bisect.insort(
            _opened,
            _successor,
            key=lambda z: z[h_val])
    )

class Board(list):
    """
    Little helper class to override __eq__  for easier `if x in []`
    """

    def __eq__(self, other) -> bool:
        return self[tiles] == other[tiles]



def parse_state(state, f = lambda x: 1, goal=GOAL):
    """
    Build a Board state from list of state,
    heuristic function *f* and goal
    """
    b = Board([
        state[:],
        goal[:],
        -1,
        f,
        state.index(9),
        None,
        0
    ])
    return b


if __name__ == "__main__":

    # all the states tested
    states = [
        parse_state([
            2, 4, 7,
            1, 5, 9,
            6, 3, 8]),
        parse_state([
            1, 2, 3,
            8, 9, 4,
            7, 6, 5]),
        parse_state([
            1, 9, 3,
            8, 2, 4,
            7, 6, 5]),
        parse_state([1,2,3,7,8,4,9,6,5]),
        parse_state([
            2,8,3,
            1,6,4,
            7,9,5
        ]),
        parse_state([4,1,3,7,9,5,8,2,6], goal=[1,2,3,4,5,6,7,8,9])
    ]

    # Dict to keep track of the stats
    stats = {
        'bfs': [],
        'dfs': [],
        'b1s manhattan': [],
        'b1s hamming': [],
        'b1s manhat_collision': [],
        'b1s permutation_inversion': [],
        'A* manhattan': [],
        'A* hamming': [],
        'A* manhat_collision': [],
        'A* permutation_inversion': [],
    }

    # run all the algorithms on all the test states
    for p in states:
        x = copy.deepcopy(p)
        res = bfs(x)
        stats['bfs'].append(res)
        print(stats)


        x = copy.deepcopy(p)
        x[h_func] = manhattan
        res = b1s(x)
        stats['b1s manhattan'].append(res)
        print(stats)


        x = copy.deepcopy(p)
        x[h_func] = hamming
        res = b1s(x)
        stats['b1s hamming'].append(res)
        print(stats)


        x = copy.deepcopy(p)
        x[h_func] = permutation_inversion
        stats['b1s permutation_inversion'].append(b1s(x))
        print(stats)


        x = copy.deepcopy(p)
        x[h_func] = manhat_collision
        stats['b1s manhat_collision'].append(b1s(x))
        print(stats)

        #############################################3
        x = copy.deepcopy(p)
        x[h_func] = lambda x: manhattan(x) + x[cost]
        res = b1s(x)
        stats['A* manhattan'].append(res)
        print(stats)


        x = copy.deepcopy(p)
        x[h_func] = lambda x: hamming(x) + x[cost]
        res = b1s(x)
        stats['A* hamming'].append(res)
        print(stats)


        x = copy.deepcopy(p)
        x[h_func] = lambda x: permutation_inversion(x) + x[cost]
        stats['A* permutation_inversion'].append(b1s(x))
        print(stats)


        x = copy.deepcopy(p)
        x[h_func] = lambda x: manhat_collision(x) + x[cost]
        stats['A* manhat_collision'].append(b1s(x))
        print(stats)

        x = copy.deepcopy(p)
        res = dfs(x)
        stats['dfs'].append(res)
        print(stats)

    # Pretty Print
    for i in stats.keys():
        print(i, stats[i])
