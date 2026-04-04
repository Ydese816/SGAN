import torch


def smoothness_v2(sequence):
    second_order_diff = sequence[2:] - 2 * sequence[1:-1] + sequence[:-2]
    smoothness_value = torch.norm(second_order_diff, p=2)

    smoothness_value = torch.sigmoid(smoothness_value)
    return smoothness_value.item()


sequence = torch.tensor([1.0, 4.0, 4.0, 7.0, 11.0])
smoothness_value = smoothness_v2(sequence)
print("smooth:", smoothness_value)
