import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import dnnlib
import tqdm

class InceptionFeatureExtractor:
    def __init__(self, device=torch.device('cuda')):
        # 加载 Inception-v3 模型
        self.device = device
        self.detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        self.detector_kwargs = dict(return_features=True)
        self.feature_dim = 2048
        
        # 加载模型
        with dnnlib.util.open_url(self.detector_url) as f:
            self.detector_net = pickle.load(f).to(device)
        self.detector_net.eval()

    def extract_features(self, images):
        """
        提取图像的 Inception 特征
        
        Args:
            images: 形状为 [B, C, H, W] 的图像张量
            
        Returns:
            features: 形状为 [B, 2048] 的特征张量
        """
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # 确保图像在正确的设备上
        images = images.to(self.device)
        
        # 提取特征
        features = self.detector_net(images, **self.detector_kwargs)
        return features

def compute_inception_mse_loss(student_images, teacher_images, feature_extractor):
    """
    计算两组图像的 Inception 特征 MSE loss
    
    Args:
        student_images: 学生模型生成的图像，形状为 [B, C, H, W]
        teacher_images: 教师模型生成的图像，形状为 [B, C, H, W]
        feature_extractor: InceptionFeatureExtractor 实例
        
    Returns:
        loss: MSE loss 标量值
    """
    # 提取特征
    student_features = feature_extractor.extract_features(student_images)
    teacher_features = feature_extractor.extract_features(teacher_images)
    
    # 计算 MSE loss
    loss = F.mse_loss(student_features, teacher_features)

    return loss