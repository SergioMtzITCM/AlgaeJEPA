import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from configs.config import MicroViTConfig

class ConvStem(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        super().__init__()

        layers = []
        curr_in = in_channels

        for _ in range(4):
            layers.extend([
                nn.Conv2d(curr_in, out_channels, kernel_size = 3, stride = 2, padding = 1, bias = False),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            ])

            curr_in = out_channels

        self.conv_blocks = nn.Sequential(*layers)

    def forward(self, 
                pixel_values: torch.Tensor) -> torch.Tensor:
        return self.conv_blocks(pixel_values)

class DWConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 init_values: float = 1e-5) -> None:
        super().__init__()

        self.gamma = nn.Parameter(init_values * torch.ones(in_channels, 1, 1))
        self.norm = nn.BatchNorm2d(in_channels)
        
        self.dwconv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            groups = in_channels
        )

    def forward(self,
                X_input: torch.Tensor) -> torch.Tensor:

        # X^ = X + λ * DWConv(Norm(X))
        return X_input + self.gamma * self.dwconv(self.norm(X_input))


class FFN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion_rate: int = 2) -> None:
        super().__init__()

        self.norm = nn.BatchNorm2d(in_channels)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * expansion_rate, kernel_size = 1), # Expansion
            nn.GELU(),
            nn.Conv2d(out_channels * expansion_rate, in_channels, kernel_size = 1) # Reduction
        )

    def forward(self,
                X_input: torch.Tensor) -> torch.Tensor:

        # FFN(X^) = σ(Norm(X^) * W_fc1) * W_fc2
        # σ: GELU
        # W_fc1: Expansion Conv
        # W_fc2: Reduction Conv
        return self.net(self.norm(X_input))

class DWStage(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion_rate: int = 2,
                 init_values: float = 1e-5) -> None:
        super().__init__()

        self.gamma = nn.Parameter(init_values * torch.ones(in_channels, 1, 1))
        
        self.DWConv = DWConvBlock(
            in_channels = in_channels,
            out_channels = out_channels,
            init_values = init_values
        )

        self.ffn = FFN(
            in_channels = in_channels,
            out_channels = out_channels,
            expansion_rate = expansion_rate
        )

    def forward(self,
                X_input: torch.Tensor) -> torch.Tensor:

        # X^
        X = self.DWConv(X_input)

        # X^^ = X^ + λ * FFN(X^)
        return X + self.gamma * self.ffn(X) 


class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1)

    def forward(self, 
                X_input: torch.Tensor) -> torch.Tensor:
        return self.conv(X_input)




class ESHA(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 sr_ratio: int) -> None:
        super().__init__()

        self.SR_ratio = sr_ratio
        self.r = 0.25
        
        self.QK_channels = 16
        self.V_channels = int((self.r * in_channels)) # rC
        self.U_channels = int((in_channels - (self.r * in_channels))) # (C - rC)
        self.input_proj_out_channels = int((self.QK_channels * 2) + self.V_channels + self.U_channels)

        self.shape_list = [self.QK_channels, self.QK_channels, self.V_channels, self.U_channels]

        # Input projection
        self.input_projection = nn.Conv2d(
            in_channels, self.input_proj_out_channels, kernel_size = 3, padding = 1, groups = 32
        )

        # Spatial reduction of K and V
        self.K_dwconv = nn.Conv2d(
            self.QK_channels, self.QK_channels, kernel_size = 3,
            padding = 1, stride = self.SR_ratio, groups = self.QK_channels
        )

        self.V_dwconv = nn.Conv2d(
            self.V_channels, self.V_channels, kernel_size = 3,
            padding = 1, stride = self.SR_ratio, groups = self.V_channels
        )

        # Final projection
        self.final_projection = nn.Conv2d(
            in_channels, out_channels, kernel_size = 1
        )


    def forward(self,
                X_input: torch.Tensor) -> torch.Tensor:

        X = self.input_projection(X_input)

        Q,K,V,U = torch.split(X, self.shape_list, dim = 1)

        U = F.gelu(U)

        # Spatial reduction
        K = self.K_dwconv(K)
        V = self.V_dwconv(V)

        B, _, H_q, W_q = Q.size()
        _, _, H_k, W_k = K.size()

        # Transformation for Scaled dot product attention (SDPA): (B, Num_Heads, Seq_Len, Channels)
        # Single Head
        Q_sdpa = Q.view(B, 1, self.QK_channels, H_q * W_q).transpose(2, 3) # (B, 1, N_q, C_q)
        K_sdpa = K.view(B, 1, self.QK_channels, H_k * W_k).transpose(2, 3) # (B, 1, N_k, C_q)
        V_sdpa = V.view(B, 1, self.V_channels, H_k * W_k).transpose(2, 3) # (B, 1, N_k, C_v)

        # Attention
        attn_out = F.scaled_dot_product_attention(Q_sdpa, K_sdpa, V_sdpa) # -> (B, 1, N_q, C_v)
        # B, C_v, H_q, W_q
        attention = attn_out.transpose(2, 3).contiguous().view(B, self.V_channels, H_q, W_q)

        # Concat U and Attention
        X = torch.cat((U, attention), dim = 1)

        # Final projection
        return self.final_projection(X)



class ESHAStage(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion_rate: int,
                 sr_ratio: int,
                 init_values: float = 1e-5) -> None:
        super().__init__()

        self.gamma_1 = nn.Parameter(init_values * torch.ones(in_channels, 1, 1))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(in_channels, 1, 1))

        self.norm = nn.BatchNorm2d(in_channels)
        
        self.esha = ESHA(
            in_channels = in_channels,
            out_channels = out_channels,
            sr_ratio = sr_ratio
        )

        self.ffn = FFN(
            in_channels = in_channels,
            out_channels = out_channels,
            expansion_rate = expansion_rate
        )

    def forward(self,
                X_input: torch.Tensor) -> torch.Tensor:

        X = X_input + self.gamma_1 * self.esha(self.norm(X_input))
        return X + self.gamma_2 * self.ffn(X)


class MicroViTModel(nn.Module):
    def __init__(self,
                 config: MicroViTConfig) -> None:
        super().__init__()

        self.config = config

        expansion_rate = 2

        if config.model == "S1":
            blocks = [2, 5, 5]
            sr_ratio = 2
            filters = [128, 256, 320]
            
        elif config.model == "S2":
            blocks = [2, 7, 5]
            sr_ratio = 2
            filters = [128, 320, 448]

        elif config.model == "S3":
            blocks = [3, 6, 6]
            sr_ratio = 1
            filters = [192, 384, 512]
        else:
            raise ValueError(f"Model must be 'S1', 'S2' or 'S3'")

        out_dim = filters[-1]
        hw = math.ceil(config.image_size / 64)

        self._out_shape = (out_dim, hw, hw)


        self.stem = ConvStem(
            in_channels = config.num_channels,
            out_channels = filters[0]
        )

        # Stage 1 - DWConv
        self.stage1 = nn.Sequential(*[
            DWStage(in_channels = filters[0], out_channels = filters[0],
                    init_values = config.layerscale_value)
        for _ in range(blocks[0])
        ])

        # Patch Embedding 1
        self.patch_embed1 = PatchEmbedding(
            in_channels = filters[0],
            out_channels = filters[1]
        )

        # Stage 2 - DWConv
        self.stage2 = nn.Sequential(*[
            DWStage(in_channels = filters[1], out_channels = filters[1],
                    init_values = config.layerscale_value)
        for _ in range(blocks[1])
        ])

        # Patch Embedding 2
        self.patch_embed2 = PatchEmbedding(
            in_channels = filters[1],
            out_channels = filters[2]
        )

        # Stage 3 - ESHA
        self.stage3 = nn.Sequential(*[
            ESHAStage(in_channels = filters[2], out_channels = filters[2],
                     expansion_rate = expansion_rate, sr_ratio = sr_ratio,
                     init_values = config.layerscale_value)
        for _ in range(blocks[2])
        ])

        # ---- Classifier ----
        self.num_classes = config.num_classes
        if self.num_classes is not None:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            out_channels = self.get_output_size()[0]
            self.classifier = nn.Linear(out_channels, self.num_classes)
        # --------------------

    def get_output_size(self) -> torch.Size:
        return self._out_shape
        

    def forward(self,
                pixel_values: torch.Tensor) -> torch.Tensor:

        X = self.stem(pixel_values)
        X = self.stage1(X)
        X = self.patch_embed1(X)
        X = self.stage2(X)
        X = self.patch_embed2(X)
        X = self.stage3(X)

        # ---- Classification ----
        if self.num_classes is not None:
            X = self.pool(X)
            X = torch.flatten(X, 1)
            X = self.classifier(X)
        # ------------------------
        return X

