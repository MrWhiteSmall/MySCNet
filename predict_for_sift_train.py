'''
基于 predict_one.py 的修改
目的    是为了预测一个数据集的所有图片
        找出预测准确的图片，copy到新的train文件夹
后续    对这个已经预测准确的图片 进行vton 然后重新预测
        筛选出 vton后  预测不准确的 或者 accuracy下降的 图片
        构建 train_vton 数据集 (表示 有效的 数据增强)
'''

import warnings
warnings.filterwarnings("ignore")

import os,shutil
import os.path as osp
import torch
from data import build_dataloader
from models import build_model
from models.img_resnet import GAP_Classifier
import data.img_transforms as T

from util_parse import parse_option

from tools.eval_metrics import evaluate_with_predidx as evaluate
from util_predict_one import extract_single_image_feature, extract_img_feature_gallery

dic = {
    'LTCC_ReID_my': 'ltcc',
    'DeepChangeDataset_my': 'deepchange',
    'prcc_my': 'prcc',
    'VCC_my': 'vcc',
}

dataset_idx = 3  # 选择数据集索引
datasets = ['LTCC_ReID_my', 'DeepChangeDataset_my', 'prcc_my', 'VCC_my']
dataset = datasets[dataset_idx] 
ckp = f'./ckp/{dic[dataset]}_best.pth.tar'
gallery_cache_name = f'./cache/{dic[dataset]}_gallery.pt'

config = parse_option(dataset)

transform_test = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
trainloader, queryloader, galleryloader, dataset, train_sampler = build_dataloader(config)
clothes2label_test = dataset.clothes2label_test
# print(clothes2label_test)

# Build model
def get_model(config,ckp,dataset):
    model, attention= build_model(config)
    checkpoint = torch.load(ckp)
    del attention
    gap_classifier = GAP_Classifier(config, dataset.num_train_pids)
    gap_classifier_h = GAP_Classifier(config, dataset.num_train_pids)
    gap_classifier_b = GAP_Classifier(config, dataset.num_train_pids)

    model.load_state_dict(checkpoint['model_state_dict'])
    gap_classifier.load_state_dict(checkpoint['gap_classifier_state_dict'])
    gap_classifier_h.load_state_dict(checkpoint['gap_classifier_h_state_dict'])
    gap_classifier_b.load_state_dict(checkpoint['gap_classifier_b_state_dict'])

    model = model.cuda(0)
    gap_classifier = gap_classifier.cuda(0)
    gap_classifier_h = gap_classifier_h.cuda(0)
    gap_classifier_b = gap_classifier_b.cuda(0)

    model.eval()
    gap_classifier.eval()
    gap_classifier_h.eval()
    gap_classifier_b.eval()
    return model, gap_classifier, gap_classifier_h, gap_classifier_b

model, gap_classifier, gap_classifier_h, gap_classifier_b = get_model(config,ckp,dataset)

gf_fixed, g_pids_fixed, g_camids_fixed, g_clothes_ids_fixed = extract_img_feature_gallery(gallery_cache_name, model, galleryloader)

def get_pred_res(model, img_path, transform,clo2id):
    global gf_fixed, g_pids_fixed, g_camids_fixed, g_clothes_ids_fixed
    # copy 一份处理
    gf, g_pids, g_camids, g_clothes_ids = gf_fixed.clone(), g_pids_fixed.clone(), g_camids_fixed.clone(), g_clothes_ids_fixed.clone()
    
    qf, q_pids, q_camids, q_clothes_ids = extract_single_image_feature(model, img_path=img_path, transform=transform,clo2id=clo2id)

    # 检查qf gf 等维度
    # print(f"Gallery feature shape: {gf.shape}, PIDs: {g_pids.shape}, CamIDs: {g_camids.shape}, Clothes IDs: {g_clothes_ids.shape}")
    # print(f"Query feature shape: {qf.shape}, PIDs: {q_pids.shape}, CamIDs: {q_camids.shape}, Clothes IDs: {q_clothes_ids.shape}")

    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m, n))
    qf, gf = qf.cuda(), gf.cuda() 
    for i in range(m):
        distmat[i] = (-torch.mm(qf[i], gf.t())).cpu()
    distmat = distmat.numpy()
    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
    cmc, mAP,pred_idx = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    
    gt_id = q_pids[0]
    pred_idx = int(pred_idx[0])

    return cmc[0],mAP,gt_id,pred_idx

'''
VCC
平均 Rank-1: 92.2%, 平均 mAP: 81.2%, 正确预测数: 1014/1020
LTCC
平均 Rank-1: 60.9%, 平均 mAP: 27.5%, 正确预测数: 445/493
PRCC
平均 Rank-1: 41.8%, 平均 mAP: 41.6%, 正确预测数: 1481/3543
Deepchange
平均 Rank-1: 57.7%, 平均 mAP: 23.2%, 正确预测数: 4815/4976
'''

def sift_all(data_root):
    from tqdm import tqdm
    # 循环每个图片,预测结果 最后做平均
    rank1_list,mAP_list,cnt_true_pred = [],[],0
    new_train_root = os.path.join(os.path.dirname(data_root), 'train_vton')
    if not osp.exists(new_train_root):
        os.makedirs(new_train_root)
    for img_name in tqdm(os.listdir(data_root)):
        imgpath = osp.join(data_root, img_name)
        if not osp.isfile(imgpath):
            continue
        rank1, mAP, gt_id, pred_idx = get_pred_res(model, imgpath, transform_test, clothes2label_test)
        rank1_list.append(rank1)
        mAP_list.append(mAP)
        if gt_id == pred_idx:
            cnt_true_pred += 1
            # copy 到新的文件夹
            shutil.copy(imgpath, new_train_root)
    print(f"平均 Rank-1: {sum(rank1_list)/len(rank1_list):.1%}, 平均 mAP: {sum(mAP_list)/len(mAP_list):.1%}, 正确预测数: {cnt_true_pred}/{len(rank1_list)}")
def sift_vcc_all():
    data_root = '/root/datasets/VCC_my/query'
    sift_all(data_root)
def sift_ltcc_all():
    data_root = '/root/datasets/LTCC_ReID_my/query'
    sift_all(data_root)
def sift_deepchange_all():
    data_root = '/root/datasets/DeepChangeDataset_my/query'
    sift_all(data_root)
def sift_prcc_all():
    data_root = '/root/datasets/prcc_my/query'
    sift_all(data_root)
  
if __name__ == '__main__':
    # sift_ltcc_all()
    # sift_deepchange_all()
    # sift_prcc_all()
    sift_vcc_all()
