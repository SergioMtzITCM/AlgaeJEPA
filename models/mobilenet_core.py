import torch
import torch.nn as nn
from torchvision import models
from configs.config import MobileNetConfig

class MobileNetModel(nn.Module):
    def __init__(self, config: MobileNetConfig) -> None:
        super().__init__()

        self.config = config
        
        # Load the base model
        model_name = config.model.lower()

        if model_name == "v2":
            model = models.mobilenet_v2()
        elif model_name == "v3" or model_name == "v3-large":
            model = models.mobilenet_v3_large()
        elif model_name == "v3-small":
            model = models.mobilenet_v3_small()
        else:
            raise ValueError(f"Model '{config.model}' is not supported. Use 'V2', 'V3', 'V3-Large' or 'V3-Small'.")

        # Extract only feature layers
        self.backbone = model.features

        # Adjust the number of channels if it is necessary
        if config.num_channels != 3:
            self._adapt_input_channels()

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
        Modifies the first convolutional layer to the new number of channels.
        """

        first_layer = self.backbone[0]
        old_conv = first_layer[0]

        if not isinstance(old_conv, nn.Conv2d):
            raise TypeError("First layer is not 'nn.Conv2d'.")

        # Make new convolutional layer
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

        first_layer[0] = new_conv

    def forward(self, 
                pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.backbone(pixel_values)

        # ---- Classification ----
        if self.num_classes is not None:
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        # ------------------------

        return x

    def get_output_size(self) -> tuple[int, int, int]:

        if self._output_shape is None:
            device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")

            dummy = torch.zeros(1, self.config.num_channels, self.config.image_size,
                                self.config.image_size, device = device)

            with torch.no_grad():
                out = self.backbone(dummy)
            self._output_shape = out.shape[1:]

        return self._output_shape



