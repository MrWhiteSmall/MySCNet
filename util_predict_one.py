import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import re

from tqdm import tqdm
def parse_image_info(img_path,clo2id):
    """
    从单张图片路径中提取 pid, camid, clothes_id 等信息
    Args:
        img_path (str): 图片路径（如 "data/query/001_2_3_0001.jpg"）
        mode (str): 'sc' 或 'cc'，决定 camid 过滤规则
    Returns:
        dict: 包含 pid, camid, clothes_id, img_path 等信息
            （若不符合条件则返回 None）
    """
    pattern = re.compile(r'(\d+)_(\d+)_(\d+)_(\d+)')
    match = pattern.search(img_path)
    if not match:
        return None

    pid, camid, clothes, _ = match.groups()
    clothes_id = clo2id[pid + clothes]
    pid, camid = int(pid), int(camid)

    return {
        'img_path': img_path,
        'pid': pid,
        'camid': camid - 1,  # 保持与现有代码一致（从0开始）
        'clothes_id': clothes_id  # 注意：此处返回原始 clothes_id，未映射到 label
    }
@torch.no_grad()
def extract_single_image_feature(model, img_path, transform,clo2id):
    """
    提取单张图片的特征和元数据
    Args:
        model: 预加载的模型
        img_path: 图片路径
        transform: 图片预处理（需与训练时一致）
        mode: 'sc' 或 'cc'
    Returns:
        dict: 包含特征、pid、camid、clothes_id 等信息
    """
    # 解析图片信息
    img_info = parse_image_info(img_path,clo2id)
    if img_info is None:
        raise ValueError(f"图片 {img_path} 不符合条件（camid 过滤）")

    # 加载并预处理图片
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).cuda()  # 添加 batch 维度

    # 提取特征
    avgpool = nn.AdaptiveAvgPool2d(1)
    feature = model(img_tensor)
    feature = avgpool(feature).view(feature.size(0), -1)

    # 水平翻转增强
    flip_tensor = torch.flip(img_tensor, [3])
    flip_feature = model(flip_tensor)
    flip_feature = avgpool(flip_feature).view(flip_feature.size(0), -1)
    feature += flip_feature
    feature = F.normalize(feature, p=2, dim=1).cpu()  # 移除 batch 维度

    # return {
    #     'feature': feature,
    #     'pid': img_info['pid'],
    #     'camid': img_info['camid'],
    #     'clothes_id': img_info['clothes_id'],
    #     'img_path': img_path
    # }
    feature = feature.unsqueeze(0)  # 添加 batch 维度以保持一致性
    # pid转为torch.tensor
    img_info['pid'] = torch.tensor(img_info['pid']).unsqueeze(0)
    img_info['camid'] = torch.tensor(img_info['camid']).unsqueeze(0)
    img_info['clothes_id'] = torch.tensor(img_info['clothes_id']).unsqueeze(0)
    return feature, img_info['pid'], img_info['camid'], img_info['clothes_id']
@torch.no_grad()
def extract_img_feature_query(model, dataloader):
    avgpool = nn.AdaptiveAvgPool2d(1)
    features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (img_paths, imgs, batch_pids, batch_camids, batch_clothes_ids) in tqdm(enumerate(dataloader)):
        flip_imgs = torch.flip(imgs, [3])
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        batch_features = model(imgs)
        batch_features = avgpool(batch_features).view(batch_features.size(0), -1)
        batch_features_flip = model(flip_imgs)
        batch_features_flip = avgpool(batch_features_flip).view(batch_features_flip.size(0), -1)
        batch_features += batch_features_flip
        batch_features = F.normalize(batch_features, p=2, dim=1)
        features.append(batch_features.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)

    features = torch.cat(features, 0)
    return features, pids, camids, clothes_ids

@torch.no_grad()
def extract_img_feature_gallery(cache_name,model, dataloader):
    if osp.exists(cache_name):
        # print(f'Loading features from {cache_name}')
        cache = torch.load(cache_name)
        return cache['features'], cache['pids'], cache['camids'], cache['clothes_ids']
    else:
        avgpool = nn.AdaptiveAvgPool2d(1)
        features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
        for batch_idx, (img_paths, imgs, batch_pids, batch_camids, batch_clothes_ids) in tqdm(enumerate(dataloader)):
            flip_imgs = torch.flip(imgs, [3])
            imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
            batch_features = model(imgs)
            batch_features = avgpool(batch_features).view(batch_features.size(0), -1)
            batch_features_flip = model(flip_imgs)
            batch_features_flip = avgpool(batch_features_flip).view(batch_features_flip.size(0), -1)
            batch_features += batch_features_flip
            batch_features = F.normalize(batch_features, p=2, dim=1)
            features.append(batch_features.cpu())
            pids = torch.cat((pids, batch_pids.cpu()), dim=0)
            camids = torch.cat((camids, batch_camids.cpu()), dim=0)
            clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)

        features = torch.cat(features, 0)
        print(f'Saving features to {cache_name}')
        # 检查文件夹是否存在,如果不存在则创建
        if not osp.exists(osp.dirname(cache_name)):
            os.makedirs(osp.dirname(cache_name))
        torch.save({
            'features': features,
            'pids': pids,
            'camids': camids,
            'clothes_ids': clothes_ids
        }, cache_name)
        return features, pids, camids, clothes_ids
