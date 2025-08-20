import yaml
from torchhydro.trainers.unified_trainer import UnifiedTester


def test_simulate_lstmfloodevents():
    with open("tests/test_uh_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    tester = UnifiedTester(config)
    obs_xr, pred_xr = tester.simulate()


if __name__ == "__main__":
    test_simulate_lstmfloodevents()
