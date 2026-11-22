# Test the value-based algorithms with PyTorch.

from argparse import Namespace
from copy import deepcopy
from xuance import get_runner
import unittest

run_mode = "benchmark"  # Choices: "train", "test", "benchmark"

running_steps = 400
eval_interval = 200
device = 'cuda:0'
args = Namespace(dl_toolbox='torch', device=device,
                 running_steps=running_steps, eval_interval=eval_interval,
                 parallels=4, buffer_size=400, batch_size=16,
                 start_training=100, test_episode=1, render=False)
env_name = "classic_control"
env_id = "CartPole-v1"


class TestValueBaseAlgo(unittest.TestCase):
    def test_c51dqn(self):
        runner = get_runner(algo="c51", env=env_name, env_id=env_id, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    def test_ddqn(self):
        runner = get_runner(algo="ddqn", env=env_name, env_id=env_id, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    def test_dqn(self):
        runner = get_runner(algo="dqn", env=env_name, env_id=env_id, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    def test_drqn(self):
        args_drqn = deepcopy(args)
        args_drqn.running_steps = 600
        args_drqn.eval_interval = 300
        args_drqn.start_training = 300
        args_drqn.buffer_size = 200
        args_drqn.batch_size = 4
        runner = get_runner(algo="drqn", env=env_name, env_id=env_id, parser_args=args_drqn)
        runner.run(mode=run_mode, running_steps=args_drqn.running_steps, eval_interval=args_drqn.eval_interval)

    def test_dueldqn(self):
        runner = get_runner(algo="dueldqn", env=env_name, env_id=env_id, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    def test_noisydqn(self):
        runner = get_runner(algo="noisydqn", env=env_name, env_id=env_id, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    def test_perdqn(self):
        runner = get_runner(algo="perdqn", env=env_name, env_id=env_id, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    def test_qrdqn(self):
        runner = get_runner(algo="qrdqn", env=env_name, env_id=env_id, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)


if __name__ == "__main__":
    unittest.main()
