# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    "*** YOUR CODE HERE ***"
    #{'alpha': 0.5, 'actionFn': <function <lambda> at 0x7fb1db35f2a8>, 'gamma': 0.9, 'epsilon': 0.3}
    if args.has_key('alpha'):
      self.alpha=args['alpha']
    if args.has_key('actionFn'):
      self.actionFn=args['actionFn']
    if args.has_key('gamma'):
      self.gamma=args['gamma']
    if args.has_key('epsilon'):
      self.epsilon=args['epsilon'] # comment out epsilon=.75, so that PacmanQAgent can imporve scores!!
    self.Q = util.Counter()

  #add these setter functions for Crawler to work
  def setEpsilon(self,epsilon):
    self.epsilon= epsilon
  def setLearningRate(self,alpha):
    self.alpha=alpha
  def setDiscount(self,gamma):
    self.gamma=gamma

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    "*** YOUR CODE HERE ***"
    if not self.Q.has_key((state,action)):
      self.Q[(state,action)]=0.0
    return self.Q[(state,action)]
    util.raiseNotDefined()


  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    a = self.best_action_from_state(state)
    return self.getQValue(state,a)
    util.raiseNotDefined()

  def best_action_from_state(self,state):
    actions = self.getLegalActions(state)
    action = None
    if len(actions)>0:
      action=max([(self.getQValue(state,a),a) for a in actions])[1]
    return action

  def best_qvalue_from_state(self,state):
    actions = self.getLegalActions(state)
    q = 0
    if len(actions)>0:
      q=max([self.getQValue(state,a) for a in actions])
    return q

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    return self.best_action_from_state(state)
    util.raiseNotDefined()

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    action = None
    "*** YOUR CODE HERE ***"
    if len(legalActions)!=0:
          if util.flipCoin(self.epsilon):
            action=random.choice(legalActions)
          else:
            action=self.getPolicy(state)
    return action
    util.raiseNotDefined()


  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    "*** YOUR CODE HERE ***"
    alpha = self.alpha
    qvalue = self.getQValue(state,action)
    
    sample = reward + self.gamma * self.best_qvalue_from_state(nextState)
    qvalue = (1-alpha) * qvalue + alpha * sample
    self.Q[(state,action)] = qvalue
    """
    if reward!=0:
      print state,action,nextState,reward
      print self.Q
      raw_input('...')
    """

    #util.raiseNotDefined()

class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action


class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      pass
