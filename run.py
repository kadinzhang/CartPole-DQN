import matplotlib
import matplotlib.pyplot as plt
import time

import tensorflow as tf
import tf_agents
import numpy as np
import PIL.Image
import pyvirtualdisplay

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.utils import common
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.policy_saver import PolicySaver
from tensorflow.keras.optimizers import Adam


if __name__ == '__main__':


    env_name = 'CartPole-v0'
    # env = suite_gym.load(env_name)
    #
    # print('Observation Spec:')
    # print(env.time_step_spec().observation)
    # print('Reward Spec:')
    # print(env.time_step_spec().reward)
    # print('Action Spec:')
    # print(env.action_spec())

    # Constants
    REPLAY_BUFFER_MAX = 100_000
    PRETRAIN_LEN = 50
    BATCH_SIZE = 64
    NUM_ITERATIONS = 20_000
    LEARNING_RATE = 1e-3

    # environment = suite_gym.load(env_name)
    # env = suite_gym.load(env_name)
    # print(env.type)

    def create_env():
        return suite_gym.load(env_name)

    parallel_env = ParallelPyEnvironment(
        [create_env] * 4
    )
    train_env = TFPyEnvironment(parallel_env)
    # train_env = TFPyEnvironment(suite_gym.load(env_name))
    eval_env = TFPyEnvironment(suite_gym.load(env_name))

    fc_layer_params = (100,)
    q_net = QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params
    )
    train_step_counter = tf.Variable(0)

    agent = DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=Adam(learning_rate=LEARNING_RATE),
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter
    )
    agent.initialize()

    random_policy = RandomTFPolicy(
        train_env.time_step_spec(),
        train_env.action_spec()
    )

    def compute_avg_return(environment, policy, num_episodes=10):
        total_return = 0
        for _ in range(num_episodes):
            time_step = environment.reset()
            episode_return = 0
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        return total_return / num_episodes

    # Replay buffer
    replay_buffer = TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=REPLAY_BUFFER_MAX
    )

    driver = DynamicStepDriver(
        train_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=1
    )

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=BATCH_SIZE,
        num_steps=2).prefetch(3)
    iterator = iter(dataset)

    agent.train_step_counter.assign(0)
    avg_return = compute_avg_return(eval_env, agent.policy)
    returns = [avg_return]

    # Pre-populate replay buffer
    for _ in range(PRETRAIN_LEN):
        driver.run()

    # Train
    # Optimize
    agent.train = common.function(agent.train)

    start_time = time.time()
    for _ in range(NUM_ITERATIONS):
        driver.run()

        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss
        step = agent.train_step_counter.numpy()

        if step % 200 == 0:
            print(f'Step {step}: loss = {train_loss}')
        if step % 1000 == 0:
            avg_return = compute_avg_return(eval_env, agent.policy)
            current_time = time.time()
            elapsed_time = current_time - start_time
            print(f'Step {step}, Time: {elapsed_time} : Average Return = {avg_return}')
            PolicySaver(agent.policy).save(f'parallel_policies/step_{step}')
            returns.append(avg_return)

    # Graph results
    iterations = range(0, NUM_ITERATIONS + 1, 1000)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.show()







