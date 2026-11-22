# Test the policy-based algorithms with TensorFlow.

from argparse import Namespace
from copy import deepcopy
import importlib.util
from xuance import get_runner
import unittest

HAS_GYM_PLATFORM = importlib.util.find_spec("gym_platform") is not None

run_mode = "benchmark"  # Choices: "train", "test", "benchmark"

running_steps = 400
eval_interval = 200
device = 'CPU'
test_mode = False
args = Namespace(dl_toolbox='tensorflow', device=device,
                 running_steps=running_steps, eval_interval=eval_interval,
                 parallels=4, horizon_size=32, n_epochs=1, n_minibatch=1,
                 buffer_size=400, batch_size=16, start_training=100,
                 test_episode=1, render=False)
env_name = "classic_control"
env_id_continuous = "Pendulum-v1"
env_id_discrete = "CartPole-v1"


class TestPolicyBaseAlgo(unittest.TestCase):

    """A2C"""
    def test_a2c_continuous(self):
        runner = get_runner(algo="a2c", env=env_name, env_id=env_id_continuous, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    def test_a2c_discrete(self):
        runner = get_runner(algo="a2c", env=env_name, env_id=env_id_discrete, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    """DDPG"""
    def test_ddpg(self):
        runner = get_runner(algo="ddpg", env=env_name, env_id=env_id_continuous, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    """P-DQN"""
    @unittest.skipUnless(HAS_GYM_PLATFORM, "gym_platform is not installed")
    def test_pdqn(self):
        runner = get_runner(algo="pdqn", env='Platform', env_id='Platform-v0', parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    """MP-DQN"""
    @unittest.skipUnless(HAS_GYM_PLATFORM, "gym_platform is not installed")
    def test_mpdqn(self):
        runner = get_runner(algo="mpdqn", env='Platform', env_id='Platform-v0', parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    """SP-DQN"""
    @unittest.skipUnless(HAS_GYM_PLATFORM, "gym_platform is not installed")
    def test_spdqn(self):
        runner = get_runner(algo="spdqn", env='Platform', env_id='Platform-v0', parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    """NPG"""
    def test_npg_continuous(self):
        runner = get_runner(algo="npg", env=env_name, env_id=env_id_continuous, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    def test_npg_discrete(self):
        runner = get_runner(algo="npg", env=env_name, env_id=env_id_discrete, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    """PG"""
    def test_pg_continuous(self):
        runner = get_runner(algo="pg", env=env_name, env_id=env_id_continuous, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    def test_pg_discrete(self):
        runner = get_runner(algo="pg", env=env_name, env_id=env_id_discrete, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    """PPG"""
    def test_ppg_continuous(self):
        args_ppg = deepcopy(args)
        args_ppg.policy_nepoch = 1
        args_ppg.value_nepoch = 1
        args_ppg.aux_nepoch = 1
        runner = get_runner(algo="ppg", env=env_name, env_id=env_id_continuous, parser_args=args_ppg)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    def test_ppg_discrete(self):
        args_ppg = deepcopy(args)
        args_ppg.policy_nepoch = 1
        args_ppg.value_nepoch = 1
        args_ppg.aux_nepoch = 1
        runner = get_runner(algo="ppg", env=env_name, env_id=env_id_discrete, parser_args=args_ppg)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    """PPO"""
    def test_ppo_continuous(self):
        runner = get_runner(algo="ppo", env=env_name, env_id=env_id_continuous, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    def test_ppo_discrete(self):
        runner = get_runner(algo="ppo", env=env_name, env_id=env_id_discrete, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    """SAC"""
    def test_sac_continuous(self):
        runner = get_runner(algo="sac", env=env_name, env_id=env_id_continuous, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    def test_sac_discrete(self):
        runner = get_runner(algo="sac", env=env_name, env_id=env_id_discrete, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    """TD3"""
    def test_td3(self):
        runner = get_runner(algo="td3", env=env_name, env_id=env_id_continuous, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)


if __name__ == "__main__":
    unittest.main()
