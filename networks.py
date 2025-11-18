import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    """
    A block of two consecutive 2D convolutions, each followed by
    Batch Normalization and a ReLU activation function.
    (CONV -> BN -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    U-Net architecture for image segmentation.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        """
        Initializes the U-Net model.

        Args:
            in_channels (int): Number of channels in the input image (e.g., 3 for RGB).
            out_channels (int): Number of channels in the output segmentation mask
                                (e.g., 1 for binary segmentation).
            features (list of int): A list specifying the number of features (channels)
                                    at each level of the U-Net encoder and decoder.
        """
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ------------------- Encoder (Contracting Path) -------------------
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # ------------------- Decoder (Expansive Path) -------------------
        for feature in reversed(features):
            # The up-convolution part
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            # The double convolution part after concatenation
            self.ups.append(DoubleConv(feature * 2, feature))

        # ------------------- Bottleneck -------------------
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # ------------------- Final Convolution -------------------
        # This final layer maps the last feature map to the desired number of output channels.
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Defines the forward pass of the U-Net.
        """
        skip_connections = []

        # --- Encoder ---
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # --- Bottleneck ---
        x = self.bottleneck(x)

        # Reverse the skip connections list for the decoder path
        skip_connections = skip_connections[::-1]

        # --- Decoder ---
        # The self.ups list contains ConvTranspose2d and DoubleConv layers alternately.
        # We iterate by steps of 2 to process them in pairs.
        for idx in range(0, len(self.ups), 2):
            # Upsample the feature map
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # If the input image size is not divisible by 16, the feature maps from the
            # skip connection and the upsampled path might have different sizes.
            # We need to resize the skip connection to match the upsampled feature map.
            if x.shape != skip_connection.shape:
                # Resize skip_connection to match x's height and width
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Concatenate the skip connection with the upsampled feature map
            concat_skip = torch.cat((skip_connection, x), dim=1)
            
            # Apply the double convolution block
            x = self.ups[idx+1](concat_skip)

        # --- Final Output ---
        return self.final_conv(x)