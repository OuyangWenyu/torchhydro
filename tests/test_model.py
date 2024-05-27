"""
Author: silencesoup silencesoup@outlook.com
Date: 2024-04-20 11:30:16
LastEditors: silencesoup silencesoup@outlook.com
LastEditTime: 2024-04-20 11:41:05
FilePath: /torchhydro/tests/test_model.py
"""

import torch
from torchhydro.models import seq2seq
from torchhydro.models import model_dict_function


def test_model():
    # put your model param here, it's just an example for seq2seq model
    model_configs = {
        "Seq2Seq": {
            "input_size": 16,
            "output_size": 1,
            "hidden_size": 256,
            "forecast_length": 24,
            "cnn_size": 120,
            "model_mode": "single",
        },
        "Seq2Seq_dual": {
            "input_size": 16,
            "output_size": 1,
            "hidden_size": 256,
            "forecast_length": 24,
            "cnn_size": 120,
            "model_mode": "dual",
            "input_size_encoder2": 1,
        },
    }
    batch_size = 2
    forecast_history = 168

    # Initialize the model
    model_config = model_configs["Seq2Seq"]

    model = model_dict_function.pytorch_model_dict["Seq2Seq"](
        input_size=model_config["input_size"],
        output_size=model_config["output_size"],
        hidden_size=model_config["hidden_size"],
        forecast_length=model_config["forecast_length"],
        cnn_size=model_config["cnn_size"],
        model_mode=model_config["model_mode"],
    )

    # Generate random inputs for testing
    # sourcery skip: no-conditionals-in-tests
    if model_config["model_mode"] == "single":
        src1 = torch.rand(
            batch_size,
            forecast_history,
            model_config["input_size"] - 1,
        )
        src2 = torch.rand(
            batch_size, forecast_history, model_config["cnn_size"]
        )
    else:  # "dual"
        src1 = torch.rand(
            batch_size, forecast_history, model_config["input_size"]
        )
        src2 = torch.rand(
            batch_size,
            forecast_history,
            model_config["input_size_encoder2"],
        )

    trg_start_token = torch.rand(batch_size, 1, model_config["output_size"])

    # Execute the model
    outputs = model(src1, src2, trg_start_token)  # Adjusted to pass parameters directly

    # Print outputs to verify
    print("Outputs shape:", outputs.shape)
    assert outputs.shape == (
        batch_size,
        model_config["forecast_length"],
        model_config["output_size"],
    ), "Output shape is incorrect"

    print("Test passed with output shape:", outputs.shape)
