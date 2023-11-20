# multiAgents.py
# --------------
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
import math
from collections import deque
from queue import Queue
from time import sleep, time

from util import manhattanDistance
from game import Directions, GameStateData
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """

    # def bfs(start_state):
    #     queue = Queue()
    #     queue.put(start_state)  # Enqueue the start state
    #     visited = set()
    #     visited.add(start_state)
    #
    #     while not queue.empty():
    #         current_state = queue.get()  # Dequeue a state from the queue
    #
    #         # Process the current state
    #         process_state(current_state)
    #
    #         # Get the next states
    #         next_states = get_next_states(current_state)
    #
    #         for next_state in next_states:
    #             if next_state not in visited:
    #                 visited.add(next_state)
    #                 queue.put(next_state)  # Enqueue the next state

    def norm_2(x1, y1, x2, y2):
        return math.sqrt((math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2)))

    def simple_nearest_food(initial_state: GameState):
        print("simple nearest food executing", end='\r')  # FIXME

        minimum1 = math.inf
        minimum2 = math.inf
        for i in range(initial_state.data.food.width):
            for j in range(initial_state.data.food.height):
                if initial_state.data.food[i][j]:
                    temp = manhattanDistance(initial_state.getPacmanPosition(), (i, j))
                    if temp < minimum1:
                        minimum1 = temp
                    temp = norm_2(*initial_state.getPacmanPosition(), i, j)
                    if temp < minimum2:
                        minimum2 = temp

        return minimum1 + minimum2

    def simple_nearest_ghost(initial_state: GameState):
        print("simple nearest ghost executing", end='\r')  # FIXME

        minimum1 = math.inf
        minimum2 = math.inf
        for i in range(initial_state.data.food.width):
            for j in range(initial_state.data.food.height):
                if (i, j) in initial_state.getGhostPositions():
                    temp = manhattanDistance(initial_state.getPacmanPosition(), (i, j))
                    if temp < minimum1:
                        minimum1 = temp
                    temp = norm_2(*initial_state.getPacmanPosition(), i, j)
                    if temp < minimum2:
                        minimum2 = temp

        return minimum1 + minimum2

    def agentDistance(initial_state: GameState, agent_index: int):
        target_pos = initial_state.getGhostPosition(agent_index)
        initial_pos = tuple(initial_state.getPacmanPosition())
        grid = initial_state.data.grid_generator()

        def myGenerateSuccessor2(position, action: str):
            x, y = position
            if action == "n":
                return tuple([x, y + 1])
            elif action == "w":
                return tuple([x - 1, y])
            elif action == "s":
                return tuple([x, y - 1])
            elif action == "e":
                return tuple([x + 1, y])

        def hasFood2(x, y):
            return grid[x][y] == '.'

        def myLegalActions2(position):
            x, y = position
            result = []
            if grid[x][y + 1] != "%":
                result.append("n")
            if grid[x - 1][y] != "%":
                result.append("w")
            if grid[x][y - 1] != "%":
                result.append("s")
            if grid[x + 1][y] != "%":
                result.append("e")
            return result

        queue = util.Queue()
        queue.push((initial_pos, 0))  # (position, distance)

        visited = set()
        visited.add(initial_pos)

        while not queue.isEmpty():
            queue = util.Queue()
            queue.push((initial_pos, 0))  # Enqueue the start state

            visited = set()
            visited.add(initial_pos)

            while not queue.isEmpty():
                current_pos, current_distance = queue.pop()

                for action in myLegalActions2(current_pos):
                    successor_pos = myGenerateSuccessor2(current_pos, action)

                    if successor_pos not in visited:
                        visited.add(successor_pos)
                        queue.push((successor_pos, current_distance + 1))

                    if successor_pos == target_pos:
                        return current_distance + 1

        return 100

    def nearestFood2(initial_state: GameState):

        initial_pos = initial_state.getPacmanPosition()
        grid = initial_state.data.grid_generator()

        def myGenerateSuccessor2(position, action: str):
            x, y = position
            if action == "n":
                return tuple([x, y + 1])
            elif action == "w":
                return tuple([x - 1, y])
            elif action == "s":
                return tuple([x, y - 1])
            elif action == "e":
                return tuple([x + 1, y])

        def hasFood2(x, y):
            return grid[x][y] == '.'

        def myLegalActions2(position):
            x, y = position
            result = []
            if grid[x][y + 1] != "%":
                result.append("n")
            if grid[x - 1][y] != "%":
                result.append("w")
            if grid[x][y - 1] != "%":
                result.append("s")
            if grid[x + 1][y] != "%":
                result.append("e")
            return result

        def hasGhost2(x, y):
            return grid[x][y] == "G"

        def hasPowerPellet2(x, y):
            return grid[x][y] == 'o'

        queue = util.Queue()
        queue.push((initial_pos, 0))  # Enqueue the start state

        visited = set()
        visited.add(initial_pos)

        while not queue.isEmpty():
            current_pos, current_distance = queue.pop()

            for action in myLegalActions2(current_pos):
                # successor_pos = myGenerateSuccessor(current_pos, data, action)
                successor_pos = myGenerateSuccessor2(current_pos, action)

                if successor_pos not in visited:
                    visited.add(successor_pos)
                    queue.push((successor_pos, current_distance + 1))
                # if hasFood(*current_pos, data):
                if hasFood2(*current_pos):
                    return current_distance + 1

        return 1

    def nearestGhost2(initial_state: GameState):
        initial_pos = initial_state.getPacmanPosition()
        grid = initial_state.data.grid_generator()

        def myGenerateSuccessor2(position, action: str):
            x, y = position
            if action == "n":
                return tuple([x, y + 1])
            elif action == "w":
                return tuple([x - 1, y])
            elif action == "s":
                return tuple([x, y - 1])
            elif action == "e":
                return tuple([x + 1, y])

        def hasFood2(x, y):
            return grid[x][y] == '.'

        def myLegalActions2(position):
            x, y = position
            result = []
            if grid[x][y + 1] != "%":
                result.append("n")
            if grid[x - 1][y] != "%":
                result.append("w")
            if grid[x][y - 1] != "%":
                result.append("s")
            if grid[x + 1][y] != "%":
                result.append("e")
            return result

        def hasGhost2(x, y):
            return grid[x][y] == "G"

        def hasPowerPellet2(x, y):
            return grid[x][y] == 'o'

        queue = util.Queue()
        queue.push((initial_pos, 0))  # Enqueue the start state

        visited = set()
        visited.add(initial_pos)

        while not queue.isEmpty():
            current_pos, current_distance = queue.pop()

            for action in myLegalActions2(current_pos):
                successor_pos = myGenerateSuccessor2(current_pos, action)

                if successor_pos not in visited:
                    visited.add(successor_pos)
                    queue.push((successor_pos, current_distance + 1))
                if hasGhost2(*current_pos):
                    return current_distance + 1

        return 1

    def hasFood(x, y, data: GameStateData):
        return data.grid_generator()[x][y] == '.'

    def hasGhost(x, y, data: GameStateData):
        return data.grid_generator()[x][y] == "G"

    def hasPowerPellet(x, y, data: GameStateData):
        return data.grid_generator()[x][y] == 'o'

    def myGenerateSuccessor(position, data: GameStateData, action: str):
        x, y = position
        temp = data.grid_generator()
        if action == "n":
            return tuple([x, y + 1])
        elif action == "w":
            return tuple([x - 1, y])
        elif action == "s":
            return tuple([x, y - 1])
        elif action == "e":
            return tuple([x + 1, y])

    def myLegalActions(position, data: GameStateData):
        x, y = position
        temp = data.grid_generator()
        result = []
        if temp[x][y + 1] != "%":
            result.append("n")
        if temp[x - 1][y] != "%":
            result.append("w")
        if temp[x][y - 1] != "%":
            result.append("s")
        if temp[x + 1][y] != "%":
            result.append("e")
        return result

    x = nearestFood2(currentGameState)
    y = nearestGhost2(currentGameState)

    total_score = currentGameState.getScore() + (1 / x ** 2) + (0.001 * y)

    # ghostStates = currentGameState.getGhostStates()
    # scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    # minimumTime = min(scaredTimes)
    # if minimumTime > 0:
    #     i = scaredTimes.index(minimumTime)
    #     t = agentDistance(currentGameState, i + 1)
    #     # total_score -= (0.001 * y) # + (1 / x ** 2)
    #     total_score += 2 / t**2

    return total_score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def getAction(self, gameState: GameState):
        def value(state: GameState, depth: int, current_index: int):
            if state.isWin() or state.isLose():
                # print(state)
                # print("score generated: ", self.evaluationFunction(state))  # TODO
                return self.evaluationFunction(state)
            elif current_index == 0:
                return MAX_value(state, depth, current_index)
            else:
                return MIN_value(state, depth, current_index)

        def MAX_value(state: GameState, depth: int, current_index: int):
            if depth == 0:
                # print(state)
                # print("score generated: ", self.evaluationFunction(state))  # TODO
                return self.evaluationFunction(state)
            v = -math.inf
            best_action = None
            for action in state.getLegalActions(agentIndex=current_index):
                successor = state.generateSuccessor(agentIndex=current_index, action=action)
                t = value(successor, depth, current_index + 1)

                if v < t:
                    v = t
                    best_action = action

            if depth == self.depth:
                return best_action

            return v

        def MIN_value(state: GameState, depth: int, current_index: int):
            v = math.inf
            for action in state.getLegalActions(agentIndex=current_index):
                successor = state.generateSuccessor(agentIndex=current_index, action=action)
                if state.getNumAgents() - current_index == 1:
                    v = min(v, value(successor, depth - 1, 0))
                else:
                    v = min(v, value(successor, depth, current_index + 1))

            return v

        # temp = {}
        # for action in gameState.getLegalActions(agentIndex=self.index):
        #     score = value(gameState.generateSuccessor(agentIndex=self.index, action=action), self.depth, 1)
        #     temp[score] = action
        #
        # answer = temp[max(temp.keys())]

        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # sleep(0.1)
        # print(gameState, end="\r")
        # print(gameState.getScore(), end="\r")
        return value(gameState, self.depth, self.index)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        alpha = -math.inf
        betta = math.inf

        def value(state: GameState, depth: int, current_index: int, alpha: float, betta: float):
            if state.isWin() or state.isLose():
                # print("score generated: ", self.evaluationFunction(state))  # TODO
                return self.evaluationFunction(state)
            elif current_index == 0:
                return MAX_value(state, depth, current_index, alpha, betta)
            else:
                return MIN_value(state, depth, current_index, alpha, betta)

        def MAX_value(state: GameState, depth: int, current_index: int, alpha: float, betta: float):
            if depth == 0:
                # print("score generated: ", self.evaluationFunction(state))  # TODO
                return self.evaluationFunction(state)
            v = -math.inf
            best_action = None
            for action in state.getLegalActions(agentIndex=current_index):
                successor = state.generateSuccessor(agentIndex=current_index, action=action)
                t = value(successor, depth, current_index + 1, alpha, betta)
                if v < t:
                    v = t
                    best_action = action

                alpha = max(alpha, v)
                if v > betta:
                    return v

            if depth == self.depth:
                return best_action

            return v

        def MIN_value(state: GameState, depth: int, current_index: int, alpha: float, betta: float):
            v = math.inf
            for action in state.getLegalActions(agentIndex=current_index):
                successor = state.generateSuccessor(agentIndex=current_index, action=action)
                if state.getNumAgents() - current_index == 1:
                    v = min(v, value(successor, depth - 1, 0, alpha, betta))
                else:
                    v = min(v, value(successor, depth, current_index + 1, alpha, betta))

                betta = min(v, betta)

                if v < alpha:
                    return v

            return v

        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return value(gameState, self.depth, self.index, alpha, betta)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here, so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
