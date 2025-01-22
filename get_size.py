import utils
import models
import yaml


# model = models.edsr.make_edsr_baseline().cuda()
# print(utils.compute_num_params(model, text=True))

# edsr-baseline 1.220M
config_pths = [
    "configs/train-lmf/train_edsr-baseline-lmnlaelte_128h_7d.yaml", # 260.6K
    # "configs/train-lmf/train_edsr-baseline-lmnlaelte_128h_6d.yaml", # 241.3K
    # "configs/train-lmf/train_edsr-baseline-lmelte_128h.yaml"
    # "configs/train-lmf/train_edsr-baseline-lmllte_96h.yaml",# 225K
    # "configs/train-lmf/train_edsr-baseline-lmllte_128h.yaml",# 271K
    # "configs/train-lmf/train_edsr-baseline-lmlte.yaml",# 271K
    # "configs/train-lmf/train_edsr-baseline-lmelte_128h_10d.yaml", #277K
    # "configs/train-lmf/train_edsr-baseline-lmnlaelte_128h.yaml", # 384.3K(concat) 281.9K(1x1conv)
    # "configs/train-lmf/train_rdn-lmelte_128h.yaml",
    # "configs/train-lmf/train_rdn-lmnlaelte_128h.yaml"
    # "configs/train-lmf/train_edsr-baseline-lmelte_192h.yaml",# 270.2K
    # "configs/train-lmf/train_edsr-baseline-lmelte_160h.yaml",#160->265.1K
    # "configs/train-lmf/train_edsr-baseline-lmelte_128h.yaml",# 128->228.4K
    # "configs/train-lmf/train_edsr-baseline-lmelte.yaml",# 256->326.4K  192->257.1K
    # "configs/train-original/train_edsr-baseline-ciaosr.yaml" #1429.1 K
    # "configs/train-lmf/train_edsr-baseline-lmnlalte.yaml", # 297.1K
    # "configs/train-original/train_edsr-baseline-elte_saa_w_nla.yaml", # 466K
    # "configs/train-lmf/train_edsr-baseline-lmciaosr.yaml", # 735.4K
    # "configs/train-original/train_edsr-baseline-slte.yaml",
    # "configs/train-lmf/train_edsr-baseline-lmelte.yaml", # 187.9K   concact 306.8K
    # "configs/train-lmf/train_edsr-baseline-lmelte_wo_nla.yaml", # 187.8K
    # "configs/train-original/train_edsr-baseline-elte_saa_wo_nla.yaml" #420.7K
    # "configs/train-original/train_edsr-baseline-elte_saa_wo_nla_w_ae.yaml",
    # "configs/train-original/train_edsr-baseline-lte.yaml", # 494.3 K +mlp_attn = 510.6K
    # "configs/train-lmf/train_edsr-baseline-lmlte.yaml", 251.9 K
    # "configs/train-original/train_edsr-baseline-elte.yaml", 420.7 K
    # "configs/train-original/train_edsr-baseline-elte_saa.yaml",
               ]
# 提取模型名称并构建字典
config_map = {}
for pth in config_pths:
    # 去掉路径前缀和文件扩展名
    base_name = pth.split('/')[-1].replace('.yaml', '')
    # 提取模型名称，假设模型名称在 'train_' 和 '.yaml' 之间
    model_name = base_name.replace('train_', '')
    config_map[model_name] = pth

for model_name,config_pth in config_map.items():
    with open(config_pth, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
    model = models.make(config['model']).cuda()
    model_size = utils.compute_num_params(model, text=False,unit='K')
    size = model_size - 1220
    print(f"Size of {model_name}: {size:.1f} K")
    
#Size of edsr-baseline-lte: 494.3 K
# config loaded.
# Size of edsr-baseline-lmlte: 251.9 K
# config loaded.
# Size of edsr-baseline-elte: 420.7 K
# config loaded.
# Size of edsr-baseline-elte_saa: 687.1 K
# 1955-1220=735k lmciaosr