import tensorflow as tf
import imageio

from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.random_tf_policy import RandomTFPolicy

env_name = 'CartPole-v0'
eval_py_env = suite_gym.load(env_name)
eval_env = TFPyEnvironment(eval_py_env)

policy_name = 'step_2000'
saved_policy = tf.saved_model.load(f'policies/{policy_name}')

def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
      time_step = eval_env.reset()
      video.append_data(eval_py_env.render())
      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        video.append_data(eval_py_env.render())


create_policy_eval_video(saved_policy, f"videos/{policy_name}")