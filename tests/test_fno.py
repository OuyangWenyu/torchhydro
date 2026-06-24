import torch
from torchhydro.models.fno import FNO


def test_fno_forward():
    # Model parameters
    n_input_features = 23
    n_output_features = 1
    modes = 16
    hidden_channels = 64
    n_layers = 4

    # Input parameters
    seq_len = 365
    batch_size = 2

    # Create model — FNO takes n_modes as a tuple
    model = FNO(
        n_modes=(modes,),
        in_channels=n_input_features,
        out_channels=n_output_features,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
    )

    # Create random input [seq_len, batch, features]
    x = torch.randn(seq_len, batch_size, n_input_features)

    # Forward pass
    with torch.no_grad():
        out = model(x)

    # Check output shape [seq_len, batch, out_channels]
    expected_shape = (seq_len, batch_size, n_output_features)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"


if __name__ == "__main__":
    test_fno_forward()
