# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, we implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util 
from util import Stack
from util import Queue 


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

"""/*=====Start Change Task 3=====*/"""


def depthFirstSearch(problem):
    # Initialize the fringe as a Stack for LIFO behavior.
    fringe = Stack()
    # Push the initial state, an empty action list, and initial depth.
    fringe.push((problem.getStartState(), [], 0))

    # Track visited states to avoid revisiting.
    exploredNodes = set()

    # Metrics for tracking the search process.
    expandedNodes = 0
    fringeSize = 0

    while not fringe.isEmpty():
        currentState, actions, depth = fringe.pop()  # Explore the next state.

        # Check goal state after popping, to ensure goal check even if state is revisited.
        if problem.isGoalState(currentState):
            return actions, expandedNodes, fringeSize, depth

        # Only proceed if we haven't explored this state yet.
        if currentState not in exploredNodes:
            exploredNodes.add(currentState)  # Mark the current state as explored.
            expandedNodes += 1

            # Limit the fringe to 500 elements.
            if fringe.size() >= 200:
                continue

            successors = problem.getSuccessors(currentState)
            for successor_State, successor_Action, _ in successors:
                # No need to check if successor_State is in exploredNodes here,
                # as we want to add all successors to the fringe for depth-first behavior.
                newActions = actions + [successor_Action]
                fringe.push((successor_State, newActions, depth + 1))
                
            fringeSize = max(fringeSize, fringe.size())  # Update the maximum fringe size observed.

    # Return defaults if goal not found.
    return [], expandedNodes, fringeSize, 0



def breadthFirstSearch(problem):
    fringe = Queue()
    exploredNodes = set()
    fringe.push((problem.getStartState(), []))
    expanded_nodes = 0
    currentFringeSize = 1  # Initialize current fringe size
    fringeSize = currentFringeSize  # Track max fringe size

    while not fringe.isEmpty():
        currentState, actions = fringe.pop()
        currentFringeSize -= 1  # Decrement for each node popped

        if currentState not in exploredNodes:
            exploredNodes.add(currentState)
            expanded_nodes += 1

            successors = problem.getSuccessors(currentState)
            for succState, succAction, _ in successors:
                if succState not in exploredNodes:
                    fringe.push((succState, actions + [succAction]))
                    currentFringeSize += 1  # Increment for each node pushed
                    fringeSize = max(fringeSize, currentFringeSize)  # Update max fringe size

            if problem.isGoalState(currentState):
                return actions, expanded_nodes, fringeSize, len(actions)

    return [], expanded_nodes, fringeSize, 0

def uniformCostSearch(problem):

    # Initialize a priority queue for the fringe to manage nodes based on the total path cost.
    fringe = util.PriorityQueue()
    # Dictionary to track the lowest cost at which we've reached each state.
    exploredNodes = {}
    startNode = (problem.getStartState(), [], 0)
    fringe.push(startNode, 0)
    # Variables to keep track of the number of nodes expanded and the maximum size of the fringe.
    expanded_nodes = 0
    fringeSize = 0

    while not fringe.isEmpty():
        fringeSize = max(fringeSize, fringe.size())
        
        # Pop the node with the lowest total path cost from the fringe.
        currentState, actions, currentCost = fringe.pop()
        
        if currentState not in exploredNodes or currentCost < exploredNodes[currentState]:
            # Record this state and its path cost as the best path found so far.
            exploredNodes[currentState] = currentCost
            expanded_nodes += 1  # Increment the count of expanded nodes.

            if problem.isGoalState(currentState):
                # If so, return the path found, number of nodes expanded, maximum fringe size, and path length.
                return actions, expanded_nodes, fringeSize, len(actions)

            # Generate successors of the current state and process each.
            for succState, succAction, succCost in problem.getSuccessors(currentState):
                newCost = currentCost + succCost  # Calculate the new total cost for reaching the successor.
                newNode = (succState, actions + [succAction], newCost)  
                
                # Add the successor to the fringe if it hasn't been explored or if this is a cheaper path.
                if succState not in exploredNodes or newCost < exploredNodes.get(succState, float('inf')):
                    fringe.update(newNode, newCost)

    return [], expanded_nodes, fringeSize, 0


"""/*=====End Change Task 3=====*/"""


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

"""/*=====Start Change Task 1=====*/"""

#heuristic for number of tiles out of place
def h1(state, problem = None):
    """
    - state: The current state of the puzzle.
    - out_of_place: The number of tiles that are misplaced.
    """

    out_of_place = 0
    goal_state = [[0,1,2], [3,4,5], [6,7,8]] #the goal state we would like to reach

    #we want to iterate through each cell in the puzzle
    for row in range(3):
        for col in range(3):
            #we will check if the value of the current cell is the same as the value of the corresponding cell in the goal state
            if state.cells[row][col] != goal_state[row][col]:
                out_of_place += 1
    
    #return the number of tiles that are out of place
    return out_of_place

#heuristic for eucleadian sum from tile to goal
def h2(state, problem = None):
     """
    - state: The current state of the puzzle.
    - eucleadian_sum : the sum of all the eucleandian distances between the placement of the tile and where it is in the goal state
    ((x2-x1)**2 + (y2-y1)**2)**0.5
    """
     eucleadian_sum = 0 #variable where we will be storing the result and return it at the end
     grid = state.cells

     #iterate over each cell of the puzzle 
     for row in range(3):
         for col in range(3):
             tile = grid[row][col]
             if tile != 0: #the tile that contains a blank space is ignored
                 eucleandian_distance = 0 #variable created in order to store the eucleadian distance in each iteration 
                 goal_row = tile // 3  #the goal row is calculated by taking the value in the cell and dividing it by 3. the returned result should be an integer
                 goal_column = tile % 3 #the goal column is calculated by taking the modulo of the value of the cell by 3
                 eucleadian_distance = ((row - goal_row) ** 2 + (col-goal_column)**2)**.5
                 eucleadian_sum += eucleadian_distance #add the distance found to the one found in the previous iterations

     return eucleadian_sum

#heuristic for manhattan sum from tile to goal
def h3(state, problem = None):
    """
    - state: The current state of the puzzle.
    - manhattan_sum : the sum of all the manhattan distances between the placement of the tile and where it is in the goal state
    abs(x2 - x1) + abs(y2 - y1)
    """
    #iterate through the puzzle 
    manhattan_sum = 0 #variable where we will be storing the result and return it at the end
    grid = state.cells
    for row in range(3):
        for col in range(3):
            tile = grid[row][col]
            if(tile != 0): #we ignore the tile that is an empty space
                manhattan_distance = 0 #variable used to store the manhnattan distance in each iteration
                goal_row = tile // 3 #the goal row is calculated by taking the value in the cell and dividing it by 3. the returned result should be an integer
                goal_column = tile % 3 #the goal column is calculated by taking the modulo of the value of the cell by 3
                manhattan_distance = abs(row-goal_row) + abs(col-goal_column)
                manhattan_sum += manhattan_distance

    return manhattan_sum

# heuristic to find the number of tiles out of their correct row and column 
def h4(state, problem = None):

    """
    - state: The current state of the puzzle.
    - total_misplaced : the sum of the tiles that are not in their goal row + the tiles that are not in their goal column
    abs(x2 - x1) + abs(y2 - y1)
    """
    total_misplaced = 0  # Initialize the total count of misplaced tiles
    for i in range(3):  # Iterate over each row
        for j in range(3):  # Iterate over each column
            current_value = state.cells[i][j]  # Get the value of the current cell
            if current_value != 0:  # Ignore the blank cell
                # Check if the current tile is misplaced in the row and column
                misplaced_row = current_value // 3 != i
                misplaced_column = current_value % 3 != j
                # Increment the count if the tile is misplaced either in row or column
                total_misplaced += int(misplaced_row) + int(misplaced_column)
    return total_misplaced  # Return the total count of misplaced tiles



def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    #to be explored (FIFO): takes in item, cost+heuristic
    frontier = util.PriorityQueue()

    exploredNodes = [] #holds (state, cost)
    
    counter_expanded=0
    fringe_len=0
    #tree_data= [0,0,0] #holds (nb_explored_nodes, fringe_len, depth)

    startState = problem.getStartState()
    startNode = (startState, [], 0) #(state, action, cost)

    frontier.push(startNode, 0)
    
    while not frontier.isEmpty():

        fringe_len= max(len(list(frontier.heap)), fringe_len)        
        #begin exploring first (lowest-combined (cost+heuristic) ) node on frontier
        currentState, actions, currentCost = frontier.pop()

        #put popped node into explored list
        currentNode = (currentState, currentCost)
        exploredNodes.append((currentState, currentCost))
        
        if problem.isGoalState(currentState): 
            return actions, counter_expanded, fringe_len, len(actions)

        else:
            #increment the expanded nodes counter
            counter_expanded+=1                    
            
            #list of (successor, action, stepCost)
            successors = problem.getSuccessors(currentState)

            #examine each successor
            for succState, succAction, succCost in successors:
                newAction = actions + [succAction]
                newCost = problem.getCostOfActions(newAction)
                newNode = (succState, newAction, newCost)

                #check if this successor has been explored
                already_explored = False
                for explored in exploredNodes:
                    #examine each explored node tuple
                    exploredState, exploredCost = explored

                    if (succState == exploredState) and (newCost >= exploredCost):
                        already_explored = True

                #if this successor not explored, put on frontier and explored list
                if not already_explored:
                    frontier.push(newNode, newCost + heuristic(succState, problem))
                    exploredNodes.append((succState, newCost))


    return actions, counter_expanded, fringe_len, len(actions)

"""/*=====End Change Task 2 =====*/"""


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
