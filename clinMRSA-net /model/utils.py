from transformers import BertTokenizer, BertModel  # 明确使用BERT类
import torch
import numpy as np
import os
import torchvision.models as models
from torchvision import transforms
from PIL import Image

import torch
import torch.nn as nn
from functools import partial
from torch.hub import load_state_dict_from_url

import timm.models.vision_transformer as vit
import timm.models.swin_transformer as swin
# import timm.models.efficientnet as effinet

from timm.models.helpers import load_state_dict


class OmniSwinTransformer(swin.SwinTransformer):
    def __init__(self, num_classes_list, projector_features=None, use_mlp=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes_list is not None

        self.projector = None
        if projector_features:
            encoder_features = self.num_features
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(nn.Linear(encoder_features, self.num_features), nn.ReLU(inplace=True),
                                               nn.Linear(self.num_features, self.num_features))
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)

        self.omni_heads = []
        for num_classes in num_classes_list:
            self.omni_heads.append(nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        self.omni_heads = nn.ModuleList(self.omni_heads)

    def forward(self, x, head_n=None):
        x = self.forward_features(x)
        if self.projector:
            x = self.projector(x)
        if head_n is not None:
            return x, self.omni_heads[head_n](x)
        else:
            return [head(x) for head in self.omni_heads]

    def generate_embeddings(self, x, after_proj=True):
        x = self.forward_features(x)
        if after_proj:
            x = self.projector(x)
        return x


class ImageEmbedder:
    _instance = None

    def __new__(cls, config):
        if cls._instance is None:
            cls._instance = super(ImageEmbedder, cls).__new__(cls)
            cls._instance.init_model(config)
        return cls._instance

    def init_model(self, config):

        num_classes_list = [14, 14, 14, 3, 6, 1]
        pretrained_weights = "../hf_models/Ark6_swinLarge768_ep50.pth.tar"
        key = "teacher"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = 768
        model = OmniSwinTransformer(num_classes_list, projector_features=1376, use_mlp=False, img_size=input_size,
                                    patch_size=4,
                                    window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
        checkpoint = torch.load(pretrained_weights, map_location=self.device,weights_only=False)
        state_dict = checkpoint[key]
        if any([True if 'module.' in k else False for k in state_dict.keys()]):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.startswith('module.')}
        k_del = []
        for k in state_dict.keys():
            if "attn_mask" in k:
                k_del.append(k)
        print(f"Removing key {k_del} from pretrained checkpoint")
        for k in k_del:
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print('Loaded with msg: {}'.format(msg))
        # 使用ResNet50作为基础模型 - 输出2048维向量
        # self.model = models.resnet50(pretrained=True)
        # 移除最后的全连接层，获取特征向量
        # self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        # self.model.eval()  # 设置为评估模式
        # 图像预处理流程
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        # 设备配置
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = self.model.to(self.device)
        # 图像向量维度
        # # 设置预处理后的图像尺寸 (3通道, 224x224)
        # self.image_shape = (3, input_size, input_size)
        # # 计算展平后的向量长度 (3*224*224 = 150528)
        # self.embedding_dim = 1376
        self.embedding_dim = 3 * 224 * 224
        self.msg_model = model.eval()
    def embed_image(self, image_path):
        """将单张图像转换为嵌入向量"""
        try:
            if not os.path.exists(image_path):
                return None

            img = Image.open(image_path).convert('RGB')
            img_tensor = self.preprocess(img)#.unsqueeze(0).to(self.device)
            # print(img_tensor.shape)
            img_tensor = img_tensor#.unsqueeze(0)
            # with torch.no_grad():
            #     embedding = self.msg_model.generate_embeddings(img_tensor)
            #     print(1111,embedding.shape)
            return img_tensor
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    def embed_images(self, image_paths, max_images=5):
        """处理一组图像路径，返回聚合后的嵌入向量"""
        if not image_paths:
            # 没有图像时返回零向量
            return np.zeros(self.embedding_dim)
        # print(len(image_paths))
        # 限制处理的图像数量以提高效率
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        # print(valid_paths)
        # if len(valid_paths) > max_images:
        #     valid_paths = valid_paths[:max_images]  # 取前几张

        embeddings = []
        for path in valid_paths:
            emb = self.embed_image(path)
            if emb is not None:
                embeddings.append(emb)
                print(emb.shape)
        if not embeddings:
            # 所有图像处理失败时返回零向量
            return np.zeros(self.embedding_dim)

        # 聚合策略：取所有图像嵌入的平均值
        return np.mean(embeddings, axis=0)


class TextEmbedder:
    def __init__(self, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)  # 直接使用BertTokenizer
        self.model = BertModel.from_pretrained(model_name)  # 直接使用BertModel
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def embed_text(self, text):
        """
        将文本切割为固定步长的块，计算每个块的 BERT 向量，并返回加权总和.
        """
        step = 256  # 步长
        max_length = 512  # 每块的最大长度
        text_chunks = [text[i:i + step] for i in range(0, len(text), step)]

        cls_vectors = []
        for chunk in text_chunks:
            # Tokenize and pad the chunk to max_length
            inputs = self.tokenizer(
                chunk,
                padding="max_length",  # Ensure padding to max_length
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extract the [CLS] vector (always at index 0)
                cls_vector = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [batch_size, hidden_size]
            cls_vectors.append(cls_vector)

        # 将所有 [CLS] 向量进行加权总和 (权重可以根据需求更改)
        cls_vectors = np.vstack(cls_vectors)  # 转为二维数组
        weights = np.ones(cls_vectors.shape[0])  # 默认权重为1
        weighted_sum = np.average(cls_vectors, axis=0, weights=weights)

        return weighted_sum
