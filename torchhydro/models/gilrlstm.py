import os
import torch
import pathlib
import cupy as cp

# Unsuccessful attempt at using cupy to implement a linear recurrence layer
kernel_path = pathlib.Path(os.getcwd()).parent.joinpath('torchhydro/models/linear_recurrence.cu')
with open(kernel_path, 'r') as kerp:
    calc_kernel = cp.RawKernel(kerp.read(), 'compute_linear_recurrence', backend='nvcc')

class GILR_LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GILR_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.output_size = output_size
        # self.fc_layer = torch.nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, x, nonlin=torch.nn.ReLU()):
        # act = self.fc_layer(x)
        # gate, impulse = torch.split(act, 2, len(act.shape) - 1)
        gate, impulse = torch.split(x, [x.shape[0] // 2, x.shape[0] // 2])
        gate = torch.sigmoid(gate.reshape(gate.shape[0], -1))
        impulse = nonlin(impulse.reshape(impulse.shape[0], -1))
        return self.linear_recurrence(gate, (1-gate) * impulse, torch.zeros(impulse.shape[1]))

    def linear_recurrence(self, decays, impulses, initial_state):
        """
        Computes the response of a linear recurrence system.

        Parameters:
        - decays: Tensor of shape (n_steps, n_dims) representing the decay coefficients.
        - impulses: Tensor of shape (n_steps, n_dims) representing the impulse inputs.
        - initial_state: Tensor of shape (n_dims,) representing the initial state of the system.

        Returns:
        - response: Tensor of shape (n_steps, n_dims) representing the response of the system.

        Example usage:
        decays = torch.tensor([[0.5, 0.6], [0.7, 0.8]], dtype=torch.float32)
        impulses = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        initial_state = torch.tensor([0.1, 0.2], dtype=torch.float32)
        response = linear_recurrence(decays, impulses, initial_state)
        """
        n_steps = impulses.shape[0]
        n_dims = impulses.shape[1]

        assert decays.shape == (n_steps, n_dims), "Decays must be a matrix of shape (n_steps, n_dims)"
        assert impulses.shape == (n_steps, n_dims), "Impulses must be a matrix of shape (n_steps, n_dims)"
        assert initial_state.shape == (n_dims,), "Initial state must be a vector of length n_dims"

        # Check if the decays tensor has the same shape as impulses
        assert decays.shape == impulses.shape, "Decay shape must match impulse shape"

        # Initialize the response tensor
        response = torch.zeros_like(impulses)

        # Compute the response
        calc_kernel((n_steps,), (n_dims,), (decays, impulses, initial_state, response, n_dims, n_steps))
        return response
