import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# Full precision model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Base vector multiplier
        base_vector = [1, 2, 2, 3, 3, 1]
        multiplier = 64

        # Define convolutional layers with batch normalization and ReLU
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=base_vector[0]*multiplier, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(base_vector[0]*multiplier)

        self.conv2 = nn.Conv2d(in_channels=base_vector[0]*multiplier, out_channels=base_vector[1]*multiplier, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(base_vector[1]*multiplier)

        self.conv3 = nn.Conv2d(in_channels=base_vector[1]*multiplier, out_channels=base_vector[2]*multiplier, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(base_vector[2]*multiplier)

        self.conv4 = nn.Conv2d(in_channels=base_vector[2]*multiplier, out_channels=base_vector[3]*multiplier, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(base_vector[3]*multiplier)

        self.conv5 = nn.Conv2d(in_channels=base_vector[3]*multiplier, out_channels=base_vector[4]*multiplier, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm2d(base_vector[4]*multiplier)

        self.conv6 = nn.Conv2d(in_channels=base_vector[4]*multiplier, out_channels=base_vector[5]*multiplier, kernel_size=1, stride=1)
        self.bn6 = nn.BatchNorm2d(base_vector[5]*multiplier)

        # Fully connected layer for classification
        self.fc = nn.Linear(base_vector[5]*multiplier, 12)  # Assuming 12 classes

    def forward(self, x):
        # Pass through convolutional layers with BatchNorm and ReLU
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))

        # Global average pooling before the fully connected layer
        x = torch.mean(x, dim=[2, 3])  # Global average pooling

        # Pass through fully connected layer
        x = self.fc(x)
        return x
    


# Custom function for binarizing weights using the straight-through estimator (STE)
class _BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp(-1, 1)

binarize_ste = _BinarizeSTE.apply

# Custom layer for binarized weights convolution
class _BinarizedWeightConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(_BinarizedWeightConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        # Binarize the weights
        binarized_weights = binarize_ste(self.conv.weight)
        # Apply the convolution using full-precision input and binarized weights
        return F.conv2d(x, binarized_weights, self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

# Define the full network using PyTorch with binarized weights
class BinarizedWeightNetwork(nn.Module):
    def __init__(self, input_shape, num_classes, base_vector=1, scaling_factors=[1, 2, 2, 3, 3]):
        super(BinarizedWeightNetwork, self).__init__()
        in_channels = input_shape[0]
        multiplier = 64

        # Layer 1: Full-precision Conv2D layer
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=base_vector * scaling_factors[0] * multiplier,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True
        )
        self.bn1 = nn.BatchNorm2d(base_vector * scaling_factors[0] * multiplier)

        # Layer 2: Binarized weights Conv2D layer
        self.conv2 = _BinarizedWeightConv2D(
            in_channels=base_vector * scaling_factors[0] * multiplier,
            out_channels=base_vector * scaling_factors[1] * multiplier,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(base_vector * scaling_factors[1] * multiplier)

        # Layer 3: Binarized weights Conv2D layer
        self.conv3 = _BinarizedWeightConv2D(
            in_channels=base_vector * scaling_factors[1] * multiplier,
            out_channels=base_vector * scaling_factors[2] * multiplier,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(base_vector * scaling_factors[2] * multiplier)

        # Layer 4: Binarized weights Conv2D layer
        self.conv4 = _BinarizedWeightConv2D(
            in_channels=base_vector * scaling_factors[2] * multiplier,
            out_channels=base_vector * scaling_factors[3] * multiplier,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn4 = nn.BatchNorm2d(base_vector * scaling_factors[3] * multiplier)

        # Layer 5: Binarized weights Conv2D layer
        self.conv5 = _BinarizedWeightConv2D(
            in_channels=base_vector * scaling_factors[3] * multiplier,
            out_channels=base_vector * scaling_factors[4] * multiplier,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.bn5 = nn.BatchNorm2d(base_vector * scaling_factors[4] * multiplier)

        # Layer 6: Final full-precision 1x1 Conv layer for N output classes
        self.conv6 = nn.Conv2d(
            in_channels=base_vector * scaling_factors[4] * multiplier,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Global Average Pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Output fully connected layer
        self.fc = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        # Forward pass through the network
        x = self.bn1(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = F.softmax(self.fc(x), dim=1)
        return x
    


# Custom clipped gradient function in PyTorch
def _clipped_gradient(x, dy, clip_value):
    """Calculate `clipped_gradient * dy`."""
    if clip_value is None:
        return dy

    # Create a mask where the gradient is only allowed for values of x <= clip_value
    mask = (x.abs() <= clip_value).float()
    return dy * mask


# Define the STE sign function using PyTorch's autograd
class _STE_Sign(Function):
    @staticmethod
    def forward(ctx, x, clip_value=1.0):
        ctx.save_for_backward(x)
        ctx.clip_value = clip_value
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        clip_value = ctx.clip_value
        # Apply the clipped gradient
        grad_input = _clipped_gradient(x, grad_output, clip_value)
        return grad_input, None


# Custom binarized convolutional layer using _STE_Sign
class _BinarizeConv2d(nn.Conv2d):
    def __init__(self, *args, clip_value=1.0, **kwargs):
        super(_BinarizeConv2d, self).__init__(*args, **kwargs)
        self.clip_value = clip_value

    def forward(self, input):
        if self.training:
            binarized_input = _STE_Sign.apply(input, self.clip_value)
        else:
            binarized_input = input.sign()

        binarized_weight = _STE_Sign.apply(self.weight, self.clip_value)
        return F.conv2d(binarized_input, binarized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# Custom network using binarized layers
class BinarizedInputNetwork(nn.Module):
    def __init__(self, input_shape, num_classes, base_vector=1, scaling_factors=[1, 2, 2, 3, 3], clip_value=1.0):
        super(BinarizedInputNetwork, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=base_vector * scaling_factors[0] * 64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True
        )
        self.bn1 = nn.BatchNorm2d(base_vector * scaling_factors[0] * 64)

        self.quant_conv2 = _BinarizeConv2d(
            in_channels=base_vector * scaling_factors[0] * 64,
            out_channels=base_vector * scaling_factors[1] * 64,
            kernel_size=3,
            stride=1,
            padding=1,
            clip_value=clip_value,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(base_vector * scaling_factors[1] * 64)

        self.quant_conv3 = _BinarizeConv2d(
            in_channels=base_vector * scaling_factors[1] * 64,
            out_channels=base_vector * scaling_factors[2] * 64,
            kernel_size=3,
            stride=2,
            padding=1,
            clip_value=clip_value,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(base_vector * scaling_factors[2] * 64)

        self.quant_conv4 = _BinarizeConv2d(
            in_channels=base_vector * scaling_factors[2] * 64,
            out_channels=base_vector * scaling_factors[3] * 64,
            kernel_size=3,
            stride=1,
            padding=1,
            clip_value=clip_value,
            bias=False
        )
        self.bn4 = nn.BatchNorm2d(base_vector * scaling_factors[3] * 64)

        self.quant_conv5 = _BinarizeConv2d(
            in_channels=base_vector * scaling_factors[3] * 64,
            out_channels=base_vector * scaling_factors[4] * 64,
            kernel_size=1,
            stride=1,
            padding=0,
            clip_value=clip_value,
            bias=False
        )
        self.bn5 = nn.BatchNorm2d(base_vector * scaling_factors[4] * 64)

        self.conv6 = nn.Conv2d(
            in_channels=base_vector * scaling_factors[4] * 64,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.quant_conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.quant_conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.quant_conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.quant_conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.softmax(x, dim=1)