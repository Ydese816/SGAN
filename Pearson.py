
import torch


def pearson_correlation(x, y):
    if len(x) != len(y):
        raise ValueError("not equal")


    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    
    covariance = torch.sum((x - mean_x) * (y - mean_y))

    std_x = torch.sqrt(torch.sum((x - mean_x)**2))
    std_y = torch.sqrt(torch.sum((y - mean_y)**2))

    if std_x == 0 or std_y == 0:
        return 0  
    else:
        pearson_coefficient = covariance / (std_x * std_y)
        return pearson_coefficient

sequence1 = torch.tensor([1., 2., 3., 4., 5.])
sequence2 = torch.tensor([90., 50., 30., 20., 10.])

result = pearson_correlation(sequence1, sequence2)
print("p:", result)

