import collections
import numpy as np
import gym
from gym import spaces
TimeStep = collections.namedtuple('TimeStep', 
        'observation, reward, is_last, info')

'''
sticker indices:
       ┌──┬──┐
       │ 0│ 1│
       ├──┼──┤
       │ 2│ 3│
 ┌──┬──┼──┼──┼──┬──┬──┬──┐
 │16│17│ 8│ 9│ 4│ 5│20│21│
 ├──┼──┼──┼──┼──┼──┼──┼──┤
 │18│19│10│11│ 6│ 7│22│23│
 └──┴──┼──┼──┼──┴──┴──┴──┘
       │12│13│
       ├──┼──┤
       │14│15│
       └──┴──┘

face colors:
    ┌──┐
    │ 0│
 ┌──┼──┼──┬──┐
 │ 4│ 2│ 1│ 5│
 └──┼──┼──┴──┘
    │ 3│
    └──┘

    ┌──┐
    │U │
 ┌──┼──┼──┬──┐
 │L │F │R │B │
 └──┼──┼──┴──┘
    │D │
    └──┘

all possible moves:
[ U , U', U2, R , R', R2, F , F', F2, D , D', D2, L , L', L2, B , B', B2, x , x', x2, y , y', y2, z , z', z2]

minimal set of moves required to solve the cube:
[ U , U', U2, R , R', R2, F , F', F2]

'''


class Rubiks2x2x2(gym.Env):

  def __init__(self,
     max_distance=3, 
     random_restarts=False, 
     distant_restarts=True, 
     random_act_prob=0.15, 
     maximize_reward=False):

    # move definitions
    self.moveDefs = np.array([ \
    [  2,  0,  3,  1, 20, 21,  6,  7,  4,  5, 10, 11, 12, 13, 14, 15,  8,  9, 18, 19, 16, 17, 22, 23], \
    [  1,  3,  0,  2,  8,  9,  6,  7, 16, 17, 10, 11, 12, 13, 14, 15, 20, 21, 18, 19,  4,  5, 22, 23], \
    [  3,  2,  1,  0, 16, 17,  6,  7, 20, 21, 10, 11, 12, 13, 14, 15,  4,  5, 18, 19,  8,  9, 22, 23], \

    [  0,  9,  2, 11,  6,  4,  7,  5,  8, 13, 10, 15, 12, 22, 14, 20, 16, 17, 18, 19,  3, 21,  1, 23], \
    [  0, 22,  2, 20,  5,  7,  4,  6,  8,  1, 10,  3, 12,  9, 14, 11, 16, 17, 18, 19, 15, 21, 13, 23], \
    [  0, 13,  2, 15,  7,  6,  5,  4,  8, 22, 10, 20, 12,  1, 14,  3, 16, 17, 18, 19, 11, 21,  9, 23], \

    [  0,  1, 19, 17,  2,  5,  3,  7, 10,  8, 11,  9,  6,  4, 14, 15, 16, 12, 18, 13, 20, 21, 22, 23], \
    [  0,  1,  4,  6, 13,  5, 12,  7,  9, 11,  8, 10, 17, 19, 14, 15, 16,  3, 18,  2, 20, 21, 22, 23], \
    [  0,  1, 13, 12, 19,  5, 17,  7, 11, 10,  9,  8,  3,  2, 14, 15, 16,  6, 18,  4, 20, 21, 22, 23], \
    ])

    # piece definitions
    self.pieceDefs = np.array([ \
      [  0, 21, 16], \
      [  2, 17,  8], \
      [  3,  9,  4], \
      [  1,  5, 20], \
      [ 12, 10, 19], \
      [ 13,  6, 11], \
      [ 15, 22,  7], \
    ])

    # OP representation from (hashed) piece stickers
    self.pieceInds = np.zeros([58, 2], dtype=np.int)
    self.pieceInds[50] = [0, 0]; self.pieceInds[54] = [0, 1]; self.pieceInds[13] = [0, 2]
    self.pieceInds[28] = [1, 0]; self.pieceInds[42] = [1, 1]; self.pieceInds[ 8] = [1, 2]
    self.pieceInds[14] = [2, 0]; self.pieceInds[21] = [2, 1]; self.pieceInds[ 4] = [2, 2]
    self.pieceInds[52] = [3, 0]; self.pieceInds[15] = [3, 1]; self.pieceInds[11] = [3, 2]
    self.pieceInds[47] = [4, 0]; self.pieceInds[30] = [4, 1]; self.pieceInds[40] = [4, 2]
    self.pieceInds[25] = [5, 0]; self.pieceInds[18] = [5, 1]; self.pieceInds[35] = [5, 2]
    self.pieceInds[23] = [6, 0]; self.pieceInds[57] = [6, 1]; self.pieceInds[37] = [6, 2]

    # useful arrays for hashing
    self.hashOP = np.array([1, 2, 10])
    self.pow3 = np.array([1, 3, 9, 27, 81, 243])
    self.fact6 = np.array([720, 120, 24, 6, 2, 1])

    self.minimal_set_of_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    self.num_actions = len(self.minimal_set_of_actions)
    self.action_space = spaces.Discrete(self.num_actions)
    self.random_act_prob = random_act_prob
    self.action_spec = DiscreteActionSpec(self.num_actions)

    self.distances = np.load('../data/distances.npy')
    self.stickers = np.load('../data/index_to_sticker.npy')
    self.action_matrices = np.load('../data/full_matrices.npy').astype(int)

    self.max_distance = max_distance
    self.num_states = len(np.where(self.distances<=self.max_distance)[0])
    self.admissible_states = np.where(self.distances<=self.max_distance)[0]
    self.most_distant_states = np.where(self.distances == self.max_distance)[0]
    self.state_idx = {}
    for idx, state in enumerate(self.admissible_states):
      self.state_idx[state] = idx
    self.admissible_scalar_stickers = self.stickers[self.admissible_states]
    print("Total states in the cube: ", len(self.admissible_states))

    num_colors = 6
    representation = []
    for i, sticker in enumerate(self.admissible_scalar_stickers):
      representation.append(np.eye(num_colors)[sticker.astype(int)].reshape(-1))
    self.admissible_obs = np.array(representation)

    self.observation_space = spaces.Discrete(self.admissible_obs[0].shape[0])
    self.random_restarts = random_restarts
    self.distant_restarts = distant_restarts
    self.maximize_reward = maximize_reward

    # pick any arbitrary state max_distance away from solution
    initial_state = self.most_distant_states[0] 
    self.initial_index = self.state_idx[initial_state]
    self.initial_stickers = self.admissible_scalar_stickers[self.initial_index]

  def reset(self):
    if self.random_restarts:
      self.current_index = np.random.randint(self.num_states)
      self.current_stickers = self.admissible_scalar_stickers[self.current_index]
    elif self.distant_restarts:
      distant_idx = self.most_distant_states[np.random.randint(len(self.most_distant_states))]
      self.current_index = self.state_idx[distant_idx]
      self.current_stickers = self.admissible_scalar_stickers[self.current_index]
    else:
      self.current_index = self.initial_index
      self.current_stickers = self.initial_stickers

    self.done = False
    self.timestep = 0
    observation = self.admissible_obs[self.current_index]

    return {'obs': observation, 
            'state': np.eye(self.num_states)[self.current_index]}

  def step(self, action):
    if np.random.rand() < self.random_act_prob:
      action = np.random.randint(self.num_actions)

    candidate_stickers = self.current_stickers[self.moveDefs[action]]
    candidate_state = self.indexOP(self.getOP(candidate_stickers.astype(int)))

    if self.distances[candidate_state] <= self.max_distance:
      self.current_index = self.state_idx[candidate_state]
      self.current_stickers = self.admissible_scalar_stickers[self.current_index]

    observation = self.admissible_obs[self.current_index]
    obs = {'obs' : observation,
           'state': np.eye(self.num_states)[self.current_index]}
    self.timestep += 1
    reward = float(self.isSolved(self.current_stickers))
    
    if self.maximize_reward:
      done = int(reward) or self.timestep == 100
    else:
      done = self.timestep == 100
    return obs, reward, done, {}

  def isSolved(self, stickers):
    for i in range(6):
      if not (stickers[4 * i:4 * i + 4] == stickers[4 * i]).all():
        return False
    return True

  # get OP representation given sticker representation (the sticker representation is FC-normalized by default)
  def getOP(self, stickers):
    return self.pieceInds[np.dot(stickers[self.pieceDefs], self.hashOP)]

  # get a unique index for the piece orientation and permutation state (0-3674159)
  def indexOP(self, sOP):
    return self.indexO(sOP) * 5040 + self.indexP2(sOP)

  # get a unique index for the piece orientation state (0-728)
  def indexO(self, sOP):
    return np.dot(sOP[:-1, 1], self.pow3)

  # get a (gap-free) unique index for the piece permutation state (0-5039)
  def indexP2(self, sOP):
    return np.dot([sOP[i, 0] - np.count_nonzero(sOP[:i, 0] < sOP[i, 0]) for i in range(6)], self.fact6)

  # print state of the cube
  def printCube(self, stickers):
    stickers = stickers.astype(int)
    print("      ┌──┬──┐")
    print("      │ {}│ {}│".format(stickers[0], stickers[1]))
    print("      ├──┼──┤")
    print("      │ {}│ {}│".format(stickers[2], stickers[3]))
    print("┌──┬──┼──┼──┼──┬──┬──┬──┐")
    print("│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│".format(stickers[16], stickers[17], stickers[8], stickers[9], stickers[4], stickers[5], stickers[20], stickers[21]))
    print("├──┼──┼──┼──┼──┼──┼──┼──┤")
    print("│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│".format(stickers[18], stickers[19], stickers[10], stickers[11], stickers[6], stickers[7], stickers[22], stickers[23]))
    print("└──┴──┼──┼──┼──┴──┴──┴──┘")
    print("      │ {}│ {}│".format(stickers[12], stickers[13]))
    print("      ├──┼──┤")
    print("      │ {}│ {}│".format(stickers[14], stickers[15]))
    print("      └──┴──┘")


def register():
  gym.envs.registration.register(
     id='Rubiks2x2x2-v0',
     entry_point='rubiks222:Rubiks2x2x2',
  )
register()

class DiscreteActionSpec:

    def __init__(self, n):
        self.n = n
    
    def sample(self):
        return np.random.randint(self.n)

    def sample_batch(self, size):
        return np.random.randint(self.n, size=size)
