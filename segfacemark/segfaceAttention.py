import warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch import Tensor, nn
import math
from typing import Tuple, Type
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Type
from torchvision.models import (
    convnext_base,
    convnext_small,
    convnext_tiny,
    swin_b,
    swin_v2_b,
    swin_v2_s,
    swin_v2_t,
    mobilenet_v3_large,
    efficientnet_v2_m,
)
import numpy as np
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class FaceDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: 256,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:

        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.background_token = nn.Embedding(1, transformer_dim)
        self.neck_token = nn.Embedding(1, transformer_dim)
        self.face_token = nn.Embedding(1, transformer_dim)
        self.cloth_token = nn.Embedding(1, transformer_dim)
        self.rightear_token = nn.Embedding(1, transformer_dim)
        self.leftear_token = nn.Embedding(1, transformer_dim)
        self.rightbro_token = nn.Embedding(1, transformer_dim)
        self.leftbro_token = nn.Embedding(1, transformer_dim)
        self.righteye_token = nn.Embedding(1, transformer_dim)
        self.lefteye_token = nn.Embedding(1, transformer_dim)
        self.nose_token = nn.Embedding(1, transformer_dim)
        self.innermouth_token = nn.Embedding(1, transformer_dim)
        self.lowerlip_token = nn.Embedding(1, transformer_dim)
        self.upperlip_token = nn.Embedding(1, transformer_dim)
        self.hair_token = nn.Embedding(1, transformer_dim)
        self.glass_token = nn.Embedding(1, transformer_dim)
        self.hat_token = nn.Embedding(1, transformer_dim)
        self.earring_token = nn.Embedding(1, transformer_dim)
        self.necklace_token = nn.Embedding(1, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )

        self.output_hypernetwork_mlps = MLP(
            transformer_dim, transformer_dim, transformer_dim // 8, 3
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        output_tokens = torch.cat(
            [
                self.background_token.weight,
                self.neck_token.weight,
                self.face_token.weight,
                self.cloth_token.weight,
                self.rightear_token.weight,
                self.leftear_token.weight,
                self.rightbro_token.weight,
                self.leftbro_token.weight,
                self.righteye_token.weight,
                self.lefteye_token.weight,
                self.nose_token.weight,
                self.innermouth_token.weight,
                self.lowerlip_token.weight,
                self.upperlip_token.weight,
                self.hair_token.weight,
                self.glass_token.weight,
                self.hat_token.weight,
                self.earring_token.weight,
                self.necklace_token.weight,
            ],
            dim=0,
        )

        tokens = output_tokens.unsqueeze(0).expand(
            image_embeddings.size(0), -1, -1
        )

        src = image_embeddings
        pos_src = image_pe.expand(image_embeddings.size(0), -1, -1, -1)
        b, c, h, w = src.shape

        hs, src = self.transformer(
            src, pos_src, tokens
        )
        mask_token_out = hs[:, :, :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(
            src
        )
        hyper_in = self.output_hypernetwork_mlps(
            mask_token_out
        )
        b, c, h, w = upscaled_embedding.shape
        seg_output = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
            b, -1, h, w
        )

        return seg_output


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class SegfaceMLP(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, 256)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class SegFaceCeleb(nn.Module):
    def __init__(self, input_resolution, model):
        super(SegFaceCeleb, self).__init__()
        self.input_resolution = input_resolution
        self.model = model

        if self.model == "swin_base":
            swin_v2 = swin_b(weights="IMAGENET1K_V1")
            self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "swinv2_base":
            swin_v2 = swin_v2_b(weights="IMAGENET1K_V1")
            self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "swinv2_small":
            swin_v2 = swin_v2_s(weights="IMAGENET1K_V1")
            self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "swinv2_tiny":
            swin_v2 = swin_v2_t(weights="IMAGENET1K_V1")
            self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "convnext_base":
            convnext = convnext_base(pretrained=False)
            self.backbone = torch.nn.Sequential(*(list(convnext.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "convnext_small":
            convnext = convnext_small(pretrained=True)
            self.backbone = torch.nn.Sequential(*(list(convnext.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "convnext_tiny":
            convnext = convnext_tiny(pretrained=True)
            self.backbone = torch.nn.Sequential(*(list(convnext.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "resnet":
            resnet101 = models.resnet101(pretrained=True)
            self.backbone = torch.nn.Sequential(*(list(resnet101.children())[:-1]))
            self.target_layer_names = ["4", "5", "6", "7"]
            self.multi_scale_features = []

        if self.model == "mobilenet":
            mobilenet = mobilenet_v3_large(pretrained=True).features
            self.backbone = mobilenet
            self.target_layer_names = ["3", "6", "12", "16"]
            self.multi_scale_features = []

        if self.model == "efficientnet":
            efficientnet = efficientnet_v2_m(pretrained=True).features
            self.backbone = efficientnet
            self.target_layer_names = ["2", "3", "5", "8"]
            self.multi_scale_features = []

        embed_dim = 1024
        out_chans = 256

        self.pe_layer = PositionEmbeddingRandom(out_chans // 2)

        for name, module in self.backbone.named_modules():
            if name in self.target_layer_names:
                module.register_forward_hook(self.save_features_hook(name))

        self.face_decoder = FaceDecoder(
            transformer_dim=256,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
        )

        num_encoder_blocks = 4
        if self.model in ["swin_base", "swinv2_base", "convnext_base"]:
            hidden_sizes = [128, 256, 512, 1024]
        if self.model in ["resnet"]:
            hidden_sizes = [256, 512, 1024, 2048]
        if self.model in [
            "swinv2_small",
            "swinv2_tiny",
            "convnext_small",
            "convnext_tiny",
        ]:
            hidden_sizes = [
                96,
                192,
                384,
                768,
            ]
        if self.model in ["mobilenet"]:
            hidden_sizes = [24, 40, 112, 960]
        if self.model in ["efficientnet"]:
            hidden_sizes = [48, 80, 176, 1280]
        decoder_hidden_size = 256

        mlps = []
        for i in range(num_encoder_blocks):
            mlp = SegfaceMLP(input_dim=hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        self.linear_fuse = nn.Conv2d(
            in_channels=decoder_hidden_size * num_encoder_blocks,
            out_channels=decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )

    def save_features_hook(self, name):
        def hook(module, input, output):
            if self.model in [
                "swin_base",
                "swinv2_base",
                "swinv2_small",
                "swinv2_tiny",
            ]:
                self.multi_scale_features.append(
                    output.permute(0, 3, 1, 2).contiguous()
                )
            if self.model in [
                "convnext_base",
                "convnext_small",
                "convnext_tiny",
                "mobilenet",
                "efficientnet",
            ]:
                self.multi_scale_features.append(
                    output
                )

        return hook

    def forward(self, x):
        self.multi_scale_features.clear()

        _, _, h, w = x.shape
        features = self.backbone(x).squeeze()

        batch_size = self.multi_scale_features[-1].shape[0]
        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(self.multi_scale_features, self.linear_c):
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(
                batch_size, -1, height, width
            )
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state,
                size=self.multi_scale_features[0].size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            all_hidden_states += (encoder_hidden_state,)

        fused_states = self.linear_fuse(
            torch.cat(all_hidden_states[::-1], dim=1)
        )
        image_pe = self.pe_layer(
            (fused_states.shape[2], fused_states.shape[3])
        ).unsqueeze(0)
        seg_output = self.face_decoder(image_embeddings=fused_states, image_pe=image_pe)

        return seg_output


def save_result(logits, output_path):
    palette = np.array(
        [
            [0, 0, 0],
            [255, 153, 51],
            [204, 0, 0],
            [0, 204, 0],
            [102, 51, 0],
            [255, 0, 0],
            [0, 255, 255],
            [255, 204, 204],
            [51, 51, 255],
            [204, 0, 204],
            [76, 153, 0],
            [102, 204, 0],
            [0, 0, 153],
            [255, 255, 0],
            [0, 0, 204],
            [204, 204, 0],
            [255, 51, 153],
            [0, 204, 204],
            [0, 51, 0],
        ],
        dtype=np.uint8,
    )

    segmentation_image = Image.fromarray(
        logits.squeeze(0).cpu().byte().numpy(), mode="P"
    )

    segmentation_image.putpalette(palette.flatten())

    segmentation_image.save(output_path)


def inference(input_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegFaceCeleb(512, "mobilenet").to(device)
    checkpoint = torch.hub.load_state_dict_from_url("https://huggingface.co/kartiknarayan/SegFace/resolve/main/mobilenet_celeba_512/model_299.pt")
    model.load_state_dict(checkpoint["state_dict_backbone"])
    model.eval()

    image = cv2.imread(input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(image)

    image = image.resize((512, 512), Image.BICUBIC)
    transforms_image_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    image = transforms_image_test(image)
    logits = model(image.unsqueeze(0).cuda())
    logits = logits.argmax(dim=1)

    save_result(logits, output_path)


def save_face_only_color_mask(mask, output_path):
    mask_np = mask[0].cpu().numpy()

    palette = np.array([
        [0, 0, 0], [255,153,51], [204,0,0], [0,204,0], [102,51,0],
        [255,0,0], [0,255,255], [255,204,204], [51,51,255],
        [204,0,204], [76,153,0], [102,204,0], [0,0,153],
        [255,255,0], [0,0,204], [204,204,0], [255,51,153],
        [0,204,204], [0,51,0]
    ], dtype=np.uint8)

    mask_rgb = palette[mask_np]

    VALID_INDICES = [2, 6, 7, 8, 9, 10, 11, 12, 13]
    mask_rgb[~np.isin(mask_np, VALID_INDICES)] = [0, 0, 0]

    cv2.imwrite(output_path, cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
    print(f"✅ 脸部七区域（含嘴唇）彩色 mask 已保存：{output_path}")

    
class SegFaceWatermarkAdapter:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = self._load_model()
        self.watermark_regions = {
            "background": {"index": 0, "weight": 2.5},   # 背景高权重：抗换脸（GAN不改背景）
            "face": {"index": 2, "weight": 0.4},        # 面部
            "lefteye": {"index": 9, "weight": 0.8},     # 左眼
            "righteye": {"index": 8, "weight": 0.8},    # 右眼
            "nose": {"index": 10, "weight": 0.8},       # 鼻子
            "mouth": {"index": 11, "weight": 0.8},      # 口腔
            "leftbro": {"index": 7, "weight": 0.4},     # 左眉
            "rightbro": {"index": 6, "weight": 0.4}     # 右眉
        }

    def _load_model(self):
        model = SegFaceCeleb(512, "mobilenet").to(self.device)

        local_model_path = "/root/autodl-tmp/SegFaceMark/model/model_299.pt"

        # checkpoint = torch.hub.load_state_dict_from_url(
        #     "https://huggingface.co/kartiknarayan/SegFace/resolve/main/mobilenet_celeba_512/model_299.pt",
        #     map_location=self.device
        # )
        checkpoint = torch.load(local_model_path, map_location=self.device)
        model.load_state_dict(checkpoint["state_dict_backbone"])
        model.eval()
        return model

    def preprocess(self, image_path, target_size=(128, 128)):
        self.original_image = cv2.imread(image_path)
        self.original_h, self.original_w = self.original_image.shape[:2]
        self.target_size = target_size
        
        image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_512 = image.resize((512, 512), Image.BICUBIC)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image_512).unsqueeze(0)
    
    def generate_attention_map(self, image_path, target_size=(128, 128)):
        image_tensor = self.preprocess(image_path, target_size)
        
        with torch.no_grad():
            logits = self.model(image_tensor.to(self.device))
            mask = logits.argmax(dim=1).squeeze().cpu().numpy()  # (512, 512)
        
        attention_map = np.zeros_like(mask, dtype=np.float32)  # (512, 512)
        
        for region in self.watermark_regions.values():
            attention_map[mask == region["index"]] = region["weight"]
        
        attention_map = cv2.resize(attention_map, target_size, interpolation=cv2.INTER_AREA)
        attention_tensor = torch.from_numpy(attention_map).unsqueeze(0).unsqueeze(0)
        
        return attention_tensor, mask
    
    @torch.no_grad()
    def generate_attention_map_tensor(self, imgs_tensor, target_size=(128, 128)):
        """
        imgs_tensor: [B, 3, H, W]  (来自dataloader，我们这里图像加载进来后处理为[-1,1])
        target_size: 生成的注意力图尺寸，可为128、256等
        Returns:
            attention_maps: [B, 1, target_size, target_size]
        """
        imgs = imgs_tensor.clone()
        imgs = imgs * 0.5 + 0.5  # [-1,1] -> [0,1]

        B = imgs.shape[0]
        imgs_512 = F.interpolate(imgs, size=(512, 512), mode='bilinear', align_corners=False)

        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for i in range(B):
            imgs_512[i] = norm(imgs_512[i])

        logits = self.model(imgs_512.to(self.device))
        masks_512 = torch.argmax(logits, dim=1).cpu().numpy()
        # print(f"生成分割掩码，shape={masks_512.shape}")

        attn_list = []
        masks_scaled_list = []
        for b in range(B):
            mask_512 = masks_512[b]  # (512,512)
            default_bg_weight = 1.2   # 给背景一个非零的基础权重
            attn = np.ones_like(mask_512, dtype=np.float32) * default_bg_weight
            # attn = np.zeros_like(mask_512, dtype=np.float32)
            for region in self.watermark_regions.values():
                attn[mask_512 == region["index"]] = region["weight"]
            attn = cv2.resize(attn, target_size, interpolation=cv2.INTER_AREA)
            attn_list.append(torch.from_numpy(attn))

            mask_scaled = cv2.resize(mask_512, target_size, interpolation=cv2.INTER_NEAREST)
            masks_scaled_list.append(torch.from_numpy(mask_scaled))

        attention_maps = torch.stack(attn_list, dim=0).unsqueeze(1).float()
        masks_scaled = torch.stack(masks_scaled_list, dim=0).long()

        # return attention_maps.to(imgs_tensor.device), masks_512[0]  # 可视化测试用
        return attention_maps.to(imgs_tensor.device), masks_scaled.to(imgs_tensor.device)




    @torch.no_grad()
    def generate_region_only_attention_map_tensor(self, imgs_tensor, target_size=(128, 128)):
        """
        仅在七个人脸语义区域内生成注意力图。
        七个区域使用原始权重，其他区域全部为 0。
        """
        imgs = imgs_tensor.clone()
        imgs = imgs * 0.5 + 0.5  # [-1,1] -> [0,1]

        B = imgs.shape[0]
        imgs_512 = F.interpolate(
            imgs, size=(512, 512), mode='bilinear', align_corners=False
        )

        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        for i in range(B):
            imgs_512[i] = norm(imgs_512[i])

        logits = self.model(imgs_512.to(self.device))
        masks_512 = torch.argmax(logits, dim=1).cpu().numpy()

        attn_list = []
        masks_scaled_list = []

        for b in range(B):
            mask_512 = masks_512[b]

            # 先用最近邻缩放 mask，保证语义类别不被插值污染
            mask_scaled = cv2.resize(
                mask_512, target_size, interpolation=cv2.INTER_NEAREST
            )

            # region-only：目标尺寸下全图先置为 0
            attn = np.zeros_like(mask_scaled, dtype=np.float32)

            # 只给七个人脸语义区域赋权重，background 和其他区域保持 0
            for name, region in self.watermark_regions.items():
                if name == "background":
                    continue
                attn[mask_scaled == region["index"]] = region["weight"]

            attn_list.append(torch.from_numpy(attn))
            masks_scaled_list.append(torch.from_numpy(mask_scaled))

        attention_maps = torch.stack(attn_list, dim=0).unsqueeze(1).float()
        masks_scaled = torch.stack(masks_scaled_list, dim=0).long()

        return attention_maps.to(imgs_tensor.device), masks_scaled.to(imgs_tensor.device)




    # def visualize_attention(self, original_image, attention_tensor, mask, output_path):
    #     attention_np = attention_tensor.squeeze().cpu().numpy()
    #     attention_colored = cv2.applyColorMap((attention_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
    #     palette = np.array([
    #         [0, 0, 0], [255, 153, 51], [204, 0, 0], [0, 204, 0], [102, 51, 0],
    #         [255, 0, 0], [0, 255, 255], [255, 204, 204], [51, 51, 255], [204, 0, 204],
    #         [76, 153, 0], [102, 204, 0], [0, 0, 153], [255, 255, 0], [0, 0, 204],
    #         [204, 204, 0], [255, 51, 153], [0, 204, 204], [0, 51, 0]
    #     ], dtype=np.uint8)
    #     mask_rgb = palette[mask] # Convert mask to RGB using the palette
    #     mask_rgb = cv2.resize(mask_rgb, (attention_np.shape[1], attention_np.shape[0]))
        
    #     original_resized = cv2.resize(original_image, (attention_np.shape[1], attention_np.shape[0]))
    #     original_rgb = cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB)
        
    #     combined = np.hstack([original_rgb, mask_rgb, attention_colored])
    #     cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    #     print(f"可视化结果已保存至：{output_path}")
    
    def visualize_attention(self, original_image, attention_tensor, mask, output_path):
        # ---------- 1️⃣ 取 batch=0 ----------
        attention_np = attention_tensor[0, 0].cpu().numpy()   # (H, W)
        mask_np = mask[0].cpu().numpy()                        # (H, W)

        # ---------- 2️⃣ 可视化 attention ----------
        attention_colored = cv2.applyColorMap(
            (attention_np * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        # ---------- 3️⃣ 语义 mask 上色 ----------
        palette = np.array([
            [0, 0, 0], [255, 153, 51], [204, 0, 0], [0, 204, 0], [102, 51, 0],
            [255, 0, 0], [0, 255, 255], [255, 204, 204], [51, 51, 255], [204, 0, 204],
            [76, 153, 0], [102, 204, 0], [0, 0, 153], [255, 255, 0], [0, 0, 204],
            [204, 204, 0], [255, 51, 153], [0, 204, 204], [0, 51, 0]
        ], dtype=np.uint8)

        mask_rgb = palette[mask_np]  # ✅ (H, W, 3)

        # ---------- 4️⃣ 尺寸对齐 ----------
        H, W = attention_np.shape
        original_resized = cv2.resize(original_image, (W, H))
        original_rgb = cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB)

        # ---------- 5️⃣ 拼接 ----------
        combined = np.hstack([original_rgb, mask_rgb, attention_colored])
        cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        print(f"✅ 可视化结果已保存至：{output_path}")




# if __name__ == "__main__":
#     # adapter = SegFaceWatermarkAdapter(device="cuda" if torch.cuda.is_available() else "cpu")
#     # attention_tensor, mask = adapter.generate_attention_map(
#     #     image_path="/home/jms/workspaces/SegFaceMark/0.jpg",
#     #     target_size=(128, 128)
#     # )
    
#     # adapter.visualize_attention(
#     #     attention_tensor, 
#     #     mask, 
#     #     output_path="watermark_attention_visualization.png"
#     # )
    
#     # print(f"注意力图形状：{attention_tensor.shape}")

#     adapter = SegFaceWatermarkAdapter(device="cuda" if torch.cuda.is_available() else "cpu")

#     image_path = "/root/autodl-tmp/SegFaceMark/ganimation.png"
#     img = Image.open(image_path).convert("RGB")
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])
#     img_tensor = transform(img).unsqueeze(0)

#     attention_tensor, mask = adapter.generate_attention_map_tensor(
#         img_tensor,
#         target_size=(128, 128)
#     )
    
#     adapter.visualize_attention(
#         cv2.imread(image_path),
#         attention_tensor, 
#         mask, 
#         output_path="watermark_attention_visualization_ganimation.png"
#     )
    
#     print(f"注意力图形状：{attention_tensor.shape}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1️⃣ 初始化 SegFace 适配器
    adapter = SegFaceWatermarkAdapter(device=device)

    # 2️⃣ 读取并预处理图像（与你当前代码一致）
    image_path = "/root/autodl-tmp/SegFaceMark/0.jpg"
    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, 128, 128]

    # 3️⃣ 生成注意力图和语义 mask
    attention_tensor, mask = adapter.generate_attention_map_tensor(
        img_tensor,
        target_size=(128, 128)
    )

    print("mask shape:", mask.shape)  # [1, 128, 128]
    print(attention_tensor)
    print(mask)



    # 4️⃣ 只保存第二列（脸部彩色，其它黑）
    # save_face_only_color_mask(
    #     mask,
    #     output_path="s_fsrt.png"
    # )