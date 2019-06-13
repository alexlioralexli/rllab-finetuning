from sandbox.finetuning.envs.mujoco.snake_env import SnakeEnv

BIG = 1e6


class DisabledSnakeEnv(SnakeEnv):
    FILE = 'snake.xml'
    ORI_IND = 2

    def step(self, action):
        action = action.copy()
        action[0] = 0.0
        return super().step(action)


