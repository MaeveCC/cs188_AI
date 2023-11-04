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


from util import manhattanDistance
from game import Directions
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        score = successorGameState.getScore()
        if newFood.count() == 0:
            return score
        
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        minFoodDistance = min(foodDistances)
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        minGhostDistance = min(ghostDistances)

        if minFoodDistance > 0:
            score += 1 / minFoodDistance
        if minGhostDistance < 2:
            score -= 2 / (minGhostDistance + 1)

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        def minimax(state, depth, agentIndex):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            legalActions = state.getLegalActions(agentIndex)
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
            
            if agentIndex == 0:  # max
                maxScore = float('-inf')
                for action in legalActions:
                    successorState = state.generateSuccessor(agentIndex, action)
                    maxScore = max(maxScore, minimax(successorState, depth, nextAgentIndex))
                return maxScore
            else:  # min
                minScore = float('inf')
                for action in legalActions:
                    successorState = state.generateSuccessor(agentIndex, action)
                    if agentIndex == state.getNumAgents() - 1:
                        minScore = min(minScore, minimax(successorState, depth + 1, nextAgentIndex))
                    else:
                        minScore = min(minScore, minimax(successorState, depth, nextAgentIndex))
                return minScore
            
        legalActions = gameState.getLegalActions(0)
        nextAgentIndex = 1
        scores = [minimax(gameState.generateSuccessor(0, action), 0, nextAgentIndex) for action in legalActions]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return legalActions[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """             
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float('-inf')
        beta = float('inf')
        maxScore = float('-inf')
        bestAction = Directions.STOP
        legalActions = gameState.getLegalActions(0).copy()
        nextAgentIndex = 1

        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            score = self.alphaBetaPruning(successorState, 0, nextAgentIndex, alpha, beta)
            if score > maxScore:
                maxScore, bestAction = score, action
            alpha = max(alpha, maxScore)
        return bestAction
    
    def alphaBetaPruning(self, state, depth, agentIndex, alpha, beta):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        legalActions = state.getLegalActions(agentIndex)
        nextAgentIndex = (agentIndex + 1) % state.getNumAgents()

        if agentIndex == 0:  # max
            maxScore = float('-inf')
            for action in legalActions:
                successorState = state.generateSuccessor(agentIndex, action)
                maxScore = max(maxScore, self.alphaBetaPruning(successorState, depth, nextAgentIndex, alpha, beta))
                if maxScore > beta:
                    return maxScore
                alpha = max(alpha, maxScore)
            return maxScore
        else:  # min
            minScore = float('inf')
            for action in legalActions:
                successorState = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    minScore = min(minScore, self.alphaBetaPruning(successorState, depth + 1, nextAgentIndex, alpha, beta))
                    if minScore < alpha:
                        return minScore
                else:
                    minScore = min(minScore, self.alphaBetaPruning(successorState, depth, nextAgentIndex, alpha, beta))
                    if minScore < alpha:
                        return minScore
                beta = min(beta, minScore)
            return minScore
                
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
        def expectimax(state, depth, agentIndex):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()

            if agentIndex == 0:  # max
                maxScore = float('-inf')
                for action in legalActions:
                    successorState = state.generateSuccessor(agentIndex, action)
                    maxScore = max(maxScore, expectimax(successorState, depth, nextAgentIndex))
                return maxScore
            else:  # expect
                expectedScore = 0
                numActions = len(legalActions)
                for action in legalActions:
                    successorState = state.generateSuccessor(agentIndex, action)
                    if agentIndex == state.getNumAgents() - 1:
                        score = expectimax(successorState, depth + 1, nextAgentIndex)
                    else:
                        score = expectimax(successorState, depth, nextAgentIndex)
                    expectedScore += score / numActions  # Uniform random choice
                return expectedScore

        legalActions = gameState.getLegalActions(0)
        nextAgentIndex = 1
        bestAction = Directions.STOP
        maxScore = float('-inf')

        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            score = expectimax(successorState, 0, nextAgentIndex)
            if score > maxScore:
                maxScore = score
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This evaluation function takes into account:
    1. Remaining food: reward states with more food left.
    2. Distance to food: reward states where Pacman is closer to food.
    3. Power pellets: reward states where power pellets are available since they allow Pacman to eat ghosts.
    4. Distance to ghosts: penalize states where Pacman is close to ghosts.
    5. Score: include the current score as a factor.
    """
    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghosts = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghosts]
    score = currentGameState.getScore()
    foodDistances = [manhattanDistance(pacmanPosition, food) for food in foods.asList()]
    ghostDistances = [manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in ghosts]

    foodReward = -len(foods.asList())
    closestFoodDistance = min(foodDistances) if foodDistances else 0
    foodDistanceReward = 1 / (closestFoodDistance + 1)
    powerPelletReward = sum(scaredTimes)
    ghostDistancePenalty = sum(ghostDistances)

    evaluationScore = (
        30 * foodReward +
        100 * foodDistanceReward +
        50 * powerPelletReward -
        20 * ghostDistancePenalty +
        score
    )

    return evaluationScore

better = betterEvaluationFunction
