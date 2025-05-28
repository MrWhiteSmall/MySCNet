import argparse
from configs.default_img import get_img_config

root_dir = '/root/datasets'
# dataset = 'VCC_my'
# dataset = 'LTCC_ReID_my'
# dataset = 'DeepChangeDataset_my'
dataset = 'prcc_my'
output_dir = './output'
is_amp = True
is_eval = False
gpu = '0'
def parse_option():
    parser = argparse.ArgumentParser(
        description='Train clothes-changing re-id model with clothes-based adversarial loss')
    # Datasets
    parser.add_argument('--root', type=str,default=root_dir, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, default=dataset, help="ltcc, prcc, vcclothes, ccvid, last, deepchange")
    # Miscs
    parser.add_argument('--output', type=str,default=output_dir, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--amp', action='store_true',default=is_amp, help="automatic mixed precision")
    parser.add_argument('--eval', action='store_true',default=is_eval, help="evaluation only")
    parser.add_argument('--gpu', default=gpu, type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args, unparsed = parser.parse_known_args()
    config = get_img_config(args) # 用arg的内容更新config

    return config