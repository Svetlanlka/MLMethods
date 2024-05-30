# environment
CONST_ENV_NAME = 'ALE/AirRaid-v5'
# path to directory with model saves
SAVES_PATH = "C:\\Users\\Mi\\my_documents\\master_program\\sem2\\MMO\code\\lab6\\model_saves"
# train step count
NUM_TIME_STEPS = 100000
# for play mode
PLAY_ROUNDS = 5

# parameters for ReplayMemory
REPLAY_CAPACITY = 1024

# parameters for DQN algorithm
BATCH_SIZE = 16
TIME_STEPS_PER_TRAIN = 1000
TRAIN_STEPS_PER_Q_SYNC = 10
TIME_STEPS_PER_SAVE = TRAIN_STEPS_PER_Q_SYNC * TIME_STEPS_PER_TRAIN * 3
MAX_NUM_TIME_STEPS = 500000
WARM_START = BATCH_SIZE * 2

REWARD_GAMMA = 0.9

INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01


def epsilon(
    i: int,
    init_ep=INITIAL_EPSILON,
    final_ep=FINAL_EPSILON,
    decrease_timesteps: int = 1000
):
    return max(
        init_ep - (init_ep - final_ep) * i / (decrease_timesteps - 1),
        final_ep
    )