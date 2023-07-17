#!/usr/bin/env python3

from collections import deque
import copy, bisect, time
from math import inf

import ipdb


grid = 0
links = 1
dim = 2
h_val = 3
h_func = 4
parent = 5
cost = 6
current_link = 7

PRINT_SOL = False



def get_moves(state):
    """
    Generate possible moves for a given Problem geometry
    """
    (width, height) = state[dim]
    return {
        "N": (-width, lambda X: X - width > 0),
        "S": (+width, lambda X: X < width * (height-1) ),
        "E": (+1, lambda X: (X % width) < width - 1),
        "W": (-1, lambda X: (X % width) > 0)
    }

def is_connected(state, number):
    """
    Verify if a given number is connected from start to end
    """
    return state[links][number][-2] is not None

def connect(state: list[list[list]], number):
    """
    Verifies if last link connect to the last point in the grid
    and creates a complete link
    """
    if not is_connected(state, number):
        width = state[dim][0]
        _link = state[links][number]
        last = to_xy(width, _link[-3])
        target = to_xy(width, _link[-1])
        diff = abs(last[0] - target[0]) + abs(last[1] - target[1])
        # Only one difference in coordinates, the link is beside target
        if diff == 1:
            _link.pop(-2)

def is_all_linked(state):
    """
    Validates if a state's links are all connecting their numbers
    """
    for i in range(len(state[links])):
        if not is_connected(state, i):
            return False
    return True

def count_empty(state):
    """
    Number of empty tiles
    """
    return len([x for x in state[grid] if x == 0])

def heuristic(state):
    """
    Wrapper around the state's heuristic function
    to check for common cases
    Invalidity is checked in the custom heuristic
    """
    if is_goal(state): h = 0
    elif is_all_linked(state) and count_empty(state) > 0:
        h = inf
    else:
        h = state[h_func](state)
    state[h_val] = h
    return h

def can_move(orig_state, D, number):
    """
    Given a move, return the new position we are going to reach
    or -1 if there is no move available

    Check if:
    - we insert a link at a valid location
    - the number we are trying to connect is not connected yet
    - the target is available
    """
    if not is_connected(orig_state, number):
        idx = orig_state[links][number][-3]
        if D[1](idx):
            new_pos = idx + D[0]
            if orig_state[grid][new_pos] == 0:
                return new_pos
    return -1

def calc_dist(a, b, width):
    """
    Helper function to compute distance between two points in the grid
    """
    (x1, y1) = to_xy(width, a)
    (x2, y2) = to_xy(width, b)
    return abs(x1-x2) + abs(y1-y2) - 1

def ko_2_admissible_h1(state):
    """
    Not in Use, for comparison and tests
    ------------------------------------
    One of the attempts to understand why A*
    is not as good as Best First
    """
    idx = state[current_link]
    w = state[dim][0]
    nc_count = sum([1 for x in state[links] if None in x])
    h = 0
    if idx is None:
        return count_empty(state)
    if is_connected(state, idx):
        for l in state[links]:
            h += calc_dist(l[-3], l[-1], w)
    _link = state[links][idx]


    nb_moves = count_moves(state, idx)
    # if not nb_moves: return inf
    h += calc_dist(_link[-3], _link[-1], w)

    empty = count_empty(state)
    # if empty - nc_count < 0: return inf
    # if empty - h < 0: return inf
    return nc_count + nb_moves


def admissible_h1(state):
    """
    Admissible Heuristic as described in Assignment #1
    """
    _links = state[links]
    w = state[dim][0]

    # Number of links not complete yet
    nc_count = 0
    h = 0
    for (idx, l) in enumerate(_links):
        if is_connected(state, idx):
            continue
        nb_moves = count_moves(state, idx)
        if not nb_moves: return inf
        nc_count += 1
        h += calc_dist(l[-3], l[-1], w)
    # return infinity if there is not enough space to link those links
    if count_empty(state) - h < 0: return inf
    if count_empty(state) - nc_count < 0: return inf
    return h



def ko_admissible_h1(state):
    """
    NOTE: not used, mainly for tests and history
        Another test to figure out why Best 1st is better

    Count number of empty spots
    and distance of each link to it's destination

    NOTE: this was modified to have better results:
        we only keeps the shortest distance and
        add the smallest number of moves
        that a link can do

    number of empty == DFS
    sum of distances == just - 1 at each level

    find which has the shortest link to
        complete and the least nb of moves available
    """
    _links = state[links]
    width = state[dim][0]
    min_moves = [inf]
    min_move = inf
    nc_count = 0
    h = 0

    chose_next = _links[0]
    dist = 0


    for (idx, l) in enumerate(_links):
        if is_connected(state, idx):
            continue
        nc_count += 1
        nb_moves = count_moves(state, idx)
        h += nb_moves
        if not nb_moves: return inf
        (x1, y1) = to_xy(width, l[-3])
        (x2, y2) = to_xy(width, l[-1])
        dist += abs(x1-x2) + abs(y1-y2) - 1

    # if dist == 9 and state[cost] == 11:
    #     ipdb.set_trace()
    if count_empty(state) - dist < 0: return inf
    if count_empty(state) - nc_count < 0: return inf
    return dist

    return min(dist, count_empty(state)) # 51 in b1st
    # that is overestimating ....
    #     nb_moves = count_moves(state, idx)
    #     h += nb_moves
    # return h
    pass


def inad_h2(state):
    """
    Inadmissible Heuristic
    NOTE: Fast but inadmissible and not working with A*

    Compute the number of possible moves each end of links
    can do to try to reach their target
    """
    _links = state[links]
    h = 0

    for (idx, l) in enumerate(_links):
        if is_connected(state, idx):
            continue
        nb_moves = count_moves(state, idx)
        if not nb_moves: return inf
        h += nb_moves
    return h

def old():
    """
    FIXME: remove, not used, bunch of tests
    """
    # trying to only return  the distance from the best link
    # but always the same anyway
    #     if nb_moves < min_move:
    #         chose_next = l
    # (x1, y1) = to_xy(width, chose_next[-3])
    # (x2, y2) = to_xy(width, chose_next[-1])
    # dist = abs(x1-x2) + abs(y1-y2) - 1

    # return dist

    # want to count the sum of the smallest moves possible
    # NOTE: doesn't work with A* ... return count_empty(state) - nc_count
    #       same:  return h/(nc_count or 1) # average
    # NOTE: fast but not admissible/consistent:
    #           it can grow when a link goes to a spot with more moves than before
    # return h/(len(_links))
    # return min(min_moves) + nc_count


    # number_empty = len([x for x in state[grid] if x == 0])
    # _links = state[links]
    # width = state[dim][0]
    # h = 0
    # min_moves = inf
    # min_h = inf
    # count_nc = 0
    # for (idx, l) in enumerate(_links):
    #     if is_connected(state, idx): continue
    #     count_nc += 1
    #     nb_moves = count_moves(state, idx)
    #     if not nb_moves: return inf
    #     min_moves = min(min_moves, nb_moves)

    #     # number_empty -= 1
    #     (x1, y1) = to_xy(width, l[-3])
    #     (x2, y2) = to_xy(width, l[-1])
    #     min_h = min(abs(x1-x2) + abs(y1-y2) - 1, min_h)
    # return number_empty + min_h
    # return min_moves + min_h
    # return abs(number_empty + min_moves)
    # return min_h + number_empty - min_moves
    pass


def calc_cost(state, orig, idx):
    """
    Helper function to compute the cost to add

    Initially here to test different cost functions
    """
    return 1

def move(orig_state, D, number):
    """
    Add a link for a given move/number if it is possible to so

    number: index of the number we are adding a link to
    D: tuple (value, function)
        - value: value to add/remove to the last link of the number selected
        - function: validates the move is possible
    """
    (w,h) = orig_state[dim]
    new_pos = can_move(orig_state, D, number)
    if new_pos > -1:
        state = copy.deepcopy(orig_state)
        # NOTE: +1 so we keep idx by 0
        state[grid][new_pos] = number + 1
        state[links][number].insert(-2, new_pos)
        connect(state, number)
        state[cost] += calc_cost(state, orig_state, number)
        state[current_link] = number
        # import ipdb;ipdb.set_trace()
        heuristic(state)
        return state
    return None

def to_xy(width,position):
    """
    Convert a 1D position in an array to the equivalent
    2D position in a grid
    returns (row, col)
    """
    return (
        position // width,
        position % width)
        # (position % WIDTH) % HEIGHT)

def to_1D(width, position):
    """
    Convert a 2D position in an grid to the equivalent
    1D position in a grid
    returns int
    """
    return width * position[0] + position[1]


def is_goal(state):
    """
    NumberLink is a solution if:
    - all the links are done
    - there is no more empty spot
    """
    no_zero = all(state[grid])
    all_linked = all([x[-2] is not None for x in state[links]])
    return no_zero and all_linked

def get_successors(state):
    """
    Generate possible successors of the state
    """
    successors = []
    s = state
    for number in range(len(state[links])):
        # TODO: keep first successor and use as base
        #       this will ensure that we only move if it's not blocking
        #       anything else
        for (_, m) in get_moves(state).items():
            s = move(state, m, number)
            if s is not None:
                successors.append(s)
    return successors

"""
Index of items in the array based on the grid position
0   1  2  3
4   5  6  7
8   9 10 11
12 13 14 15
"""

class Problem(list):
    """
    Simple class to slightly modify representation and
    equality on lists
    """
    def __eq__(self, other) -> bool:
        return self[grid] == other[grid]

    def __repr__(self) -> str:
        (w, h) = self[dim]
        _s = "\n"
        for i in range(h):
            _s += f"{(self[grid][i*w:(i*w)+w])}\n"
        return _s

def count_moves(state, idx):
    """
    Count the number of moves that a link can do
    """
    # Verify that at least one move is possible
    moves = []
    for (_, m) in get_moves(state).items():
        if can_move(state, m, idx) > -1:
            moves.append(True)
    return len(moves)


def parse_state(state, width, height):
    """
    Create a NumberLink problem from a state and dimensions
    """
    numbers = {x for x in state if x != 0}
    _grid = state
    _links = [[] for x in numbers]
    for (idx, val) in enumerate(state):
        if val != 0:
            _links[val-1].append(idx)
            if len(_links[val-1]) == 2:
                _links[val-1].insert(1, None)

    p = Problem([
        _grid,
        _links,
        (width, height),
        inf, # h_val
        lambda x: 1, # h_func
        None, # Parent
        0, # Cost
        None, #Current link
    ])
    for i in range(len(_links)):
        if len(_links[i]) < 2:
            _links[i].append(_links[i][0])
        connect(p, i)
    heuristic(p)
    return p




def trackback_solution(closed):
    """
    From the closed list, move up to each parents
    to return the complete set of moves used to
    get to the solution
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
        popfunc=lambda x: x.pop(),
        sortfunc=lambda x,y: x.appendleft(y)):
    """
    Default to BFS, takes a function to chose how to pop values
    from the open list
    and a sortfunc to define how to sort new items in the open list
    """
    global PRINT_SOL
    opened = deque()
    closed = deque()
    opened.appendleft(state)
    found = False
    time_in = time.perf_counter()
    while not found and len(opened) != 0:
        current = popfunc(opened)
        closed.append(current)
        if is_goal(current):
            found = True
            sol = trackback_solution(closed)
            time_out = time.perf_counter()
            print(f"{len(closed)} | {(time_out - time_in)*1000}|{len(sol)}|")
            if PRINT_SOL:
                print("Solution found")
                print(current)
                print(f"closed: {len(closed)}")
                print(
                    f"solution path length: {len(sol)}")
                print(f"Time to solve: {(time_out - time_in)*1000}ms")

                print(sol)
            break

        for s in get_successors(current):
            # import ipdb;ipdb.set_trace()
            # print(f"{s}, {s[h_val]}, {s[cost]}")
            if s[h_val] != inf and s not in opened and s not in closed:
                s[parent] = len(closed) - 1
                sortfunc(opened, s)
    return found

def bfs(state):
    return generic_search(state)

def b1s(state):
    """
    basically BFS and inplace sort for new elements
    based on their heuristic value
    NOTE: The heuristic is given by the state
    """
    return generic_search(
        state,
        popfunc=lambda x: x.popleft(),
        sortfunc=lambda x,y: bisect.insort(
            x, y, key=lambda z: z[h_val]
        )
    )

def inadmissible_h2(state):
    """
    Example of inadmissible heuristic for Best First
    """
    return admissible_h1(state) + count_empty(state)

def admissible_astar(state):
    """
    Admissible heuristic for A*
    """
    return admissible_h1(state) + state[cost]

def inadmissible_astar(state):
    """
    Inadmissible heuristic for A*
    """
    return inad_h2(state) + state[cost]


initial = parse_state(
    [3,0,0,2, # Grid
     0,2,1,0,
     0,0,3,1,
     4,0,0,4],
    4,4
)

if __name__ == "__main__":
    # PRINT_SOL = True

    # Dict of different states tested
    plist = {
        't1': parse_state(
            [3,0,0,2, # Grid
             0,2,1,0,
             0,0,3,1,
             4,0,0,4],
            4,4
        ),
        't2': parse_state(
            [3,0,0,2, # Grid
             0,0,0,0,
             0,2,1,0,
             0,0,3,1,
             4,0,0,4],
            4,5
        ),
        't3': parse_state(
            [
                1,2,0,0,2,
                0,0,3,0,4,
                0,0,4,0,5,
                0,0,0,3,0,
                0,0,0,1,5
            ],
            5,5),
        # multiple solutions
        't4': parse_state(
            [
                1,0,0,0,2,
                0,2,4,0,4,
                0,3,0,0,5,
                0,0,0,3,0,
                0,0,0,1,5
            ],
            5,5),
        't5': parse_state(
            [
                1,0,0,0,0,      # 1,2,2,2,2,
                0,2,0,3,2,      # 0,2,3,3,2,
                0,0,0,0,4,      # 0,3,3,4,4,
                0,0,4,0,5,      # 0,3,4,4,5,
                0,0,0,3,0,      # 0,3,3,3,5,
                0,0,0,1,5       # 0,0,0,1,5
            ],
            5,6),
        't6': parse_state(
            [1,2,3,4,
            2,0,0,0,
            3,0,0,0,
            4,0,0,0],
            4,4
        ),
        't7': parse_state(
            [1,2,0,0,
            0,1,3,0,
            2,3,0,0,
            0,0,0,0],
            4,4)}

    # Run the tests
    for (k,v) in plist.items():
        if k in ['t1', 't2', 't6', 't7']:
            print(k)
            print("bfs")
            x = copy.deepcopy(v)
            heuristic(x)
            bfs(x)
            print("|")


    for (k,v) in plist.items():
        print(k)
        print("b1s admissible")
        x = copy.deepcopy(v)
        x[h_func] = admissible_h1
        heuristic(x)
        b1s(x)

    for (k,v) in plist.items():
        if k in ['t4', 't5']:
            continue
        print(k)
        print("a* admissible")
        x = copy.deepcopy(v)
        x[h_func] = admissible_astar
        heuristic(x)
        b1s(x)

    for (k,v) in plist.items():
        print(k)
        print("b1s inadmissible")
        x = copy.deepcopy(v)
        x[h_func] = inad_h2
        heuristic(x)
        b1s(x)


    for (k,v) in plist.items():
        if k in ['t4', 't5']:
            continue
        print(k)
        print("a* inadmissible")
        x = copy.deepcopy(v)
        x[h_func] = inadmissible_astar
        heuristic(x)
        b1s(x)
