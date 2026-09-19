import torch
import torch.nn as nn
from torchvision import models
from configs.config import ResNetConfig  # Cambiamos a una config apropiada

class ResNetModel(nn.Module):
    def __init__(self, config: ResNetConfig) -> None:
        super().__init__()

        self.config = config

        # Load ResNet18 without pretrained weights (random initialization)
        model = models.resnet18()

        # Extract all layers except the final average pooling and fully connected
        # This gives a feature map of shape [batch, 512, H/32, W/32]
        self.backbone = nn.Sequential(*list(model.children())[:-2])

        # Adjust the number of input channels if necessary
        if config.num_channels != 3:
            self._adapt_input_channels()

        # Compute the number of output channels from the backbone
        self.num_features = self.get_output_size()[0]

        # Pooling & Classifier Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.num_features, config.num_classes)

        self._output_shape = None

        # ---- Classifier ----
        self.num_classes = config.num_classes
        if self.num_classes is not None:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            out_channels = self.get_output_size()[0]
            self.classifier = nn.Linear(out_channels, self.num_classes)
        # --------------------

    def _adapt_input_channels(self):
        """
        Modifies the first convolutional layer to accept a different number of input channels.
        For ResNet18, the first layer is `conv1` (a single Conv2d).
        """
        old_conv = self.backbone[0]
        if not isinstance(old_conv, nn.Conv2d):
            raise TypeError("First layer is not 'nn.Conv2d'.")

        # Create a new convolutional layer with the required number of input channels
        new_conv = nn.Conv2d(
            in_channels = self.config.num_channels,
            out_channels = old_conv.out_channels,
            kernel_size = old_conv.kernel_size,
            stride = old_conv.stride,
            padding = old_conv.padding,
            dilation = old_conv.dilation,
            groups = old_conv.groups,
            bias = old_conv.bias is not None,
            padding_mode = old_conv.padding_mode
        )
        self.backbone[0] = new_conv

    def forward(self, 
                pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.backbone(pixel_values)   # [B, 512, H/32, W/32]

        # ---- Classification ----
        if self.num_classes is not None:
            x = self.pool(x) # [B, 512, 1, 1]
            x = torch.flatten(x, 1) # [B, 512]
            x = self.classifier(x) # [B, num_classes]
        # ------------------------

        return logits

    def get_output_size(self) -> tuple[int, int, int]:
        """
        Returns the shape (channels, height, width) of the feature map
        produced by the backbone for the configured input size.
        """
        if self._output_shape is None:
            device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
            dummy = torch.zeros(1, self.config.num_channels, self.config.image_size,
                                self.config.image_size, device = device)
            with torch.no_grad():
                out = self.backbone(dummy)
            self._output_shape = out.shape[1:]  # (C, H, W)
        return self._output_shape
