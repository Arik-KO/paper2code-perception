import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert, generalized_box_iou_loss


if __name__ == "__main__":
    print("working well")
