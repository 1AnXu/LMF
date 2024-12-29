import utils
import models
import yaml


# model = models.edsr.make_edsr_baseline().cuda()
# print(utils.compute_num_params(model, text=True))

# edsr-baseline 1.220M

config_pth = "configs/train-original/train_edsr-baseline-liif.yaml"
with open(config_pth, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print('config loaded.')
model = models.make(config['model']).cuda()
model_size = utils.compute_num_params(model, text=False,unit='K')
result = model_size - 1220
print(f"{result:.1f} K")