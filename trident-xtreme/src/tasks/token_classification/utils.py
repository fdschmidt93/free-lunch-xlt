import torch

def get_last_hidden_state(self, outputs: dict, *args, **kwargs):
    outputs["hidden_states"] = outputs["hidden_states"][-1]
    return outputs
