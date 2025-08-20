import yaml
from torchhydro.trainers.resulter import Resulter
from torchhydro.trainers.unified_trainer import UnifiedTester


def test_simulate_lstmfloodevents():
    with open("tests/test_uh_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    tester = UnifiedTester(config)
    pred_xr, obs_xr = tester.simulate()
    resulter = Resulter(config)
    resulter.save_cfg(config)
    resulter.save_result(pred_xr, obs_xr)
    resulter.eval_result(pred_xr, obs_xr)


if __name__ == "__main__":
    test_simulate_lstmfloodevents()
