import torch.hub
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
from torchvision.models.video.resnet import VideoResNet, BasicBlock, R2Plus1dStem, Conv2Plus1D


model_urls = {
    "r2plus1d_34_8_ig65m": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ig65m_from_scratch-9bae36ae.pth",
    "r2plus1d_34_32_ig65m": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ig65m_from_scratch-449a7af9.pth",
    "r2plus1d_34_8_kinetics": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ft_kinetics_from_ig65m-0aa0550b.pth",
    "r2plus1d_34_32_kinetics": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth",
}


class VideoResNetWithFeatureReturn(VideoResNet):
    def __init__(self,block,conv_makers,layers,stem):
        super().__init__(block=block,conv_makers=conv_makers,layers=layers,stem=stem)
        self.pool_spatial = Reduce("n c t h w -> n c t", reduction="mean")
        self.pool_temporal = Reduce("n c t -> n c", reduction="mean")
    
    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool_spatial(x)
        x = self.pool_temporal(x)
        # x = self.avgpool(x)
        # Flatten the layer to fc
        # print(x.size())
        # x = x.flatten(1)
        x = self.fc(x)

        return x


def r2plus1d_34(pretrain=None):
    model = VideoResNetWithFeatureReturn(block=BasicBlock,
                                         conv_makers=[Conv2Plus1D] * 4,
                                         layers=[3, 4, 6, 3],
                                         stem=R2Plus1dStem)

    # model = VideoResNet(block=BasicBlock,conv_makers=[Conv2Plus1D] * 4,layers=[3, 4, 6, 3],stem=R2Plus1dStem)

    model.fc = nn.Linear(model.fc.in_features, out_features=400)

    # Fix difference in PyTorch vs Caffe2 architecture
    # https://github.com/facebookresearch/VMZ/issues/89
    # https://github.com/pytorch/vision/issues/1265
    model.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    model.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    model.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9
    if pretrain:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[pretrain],
                                                        progress=True)
        model.load_state_dict(state_dict)
        # print()

    return model

def get_vmz_fine_tuning_parameters(model, tune_last_k_layer):
    layer_max_index = 4
    if tune_last_k_layer == layer_max_index:
        return model.parameters()

    ft_begin_index = layer_max_index - tune_last_k_layer + 1

    ft_module_names = []
    for i in range(ft_begin_index, layer_max_index + 1):
        ft_module_names.append('module.base_model.layer{}'.format(i))
        ft_module_names.append('module.base_model.stem.{}'.format(i))
    ft_module_names.append('module.base_model.fc')

    parameters = []
    freeze = []
    tune = []
    names = []
    # for k, v in model.named_parameters():
    #     print(k)
    # raise RuntimeError("stop")
    for k, v in model.named_parameters():
        names.append(k)
        no_grad = True
        for ft_module in ft_module_names:
            if k.startswith(ft_module):
                parameters.append({'params': v})
                tune.append(k)
                no_grad = False
                break
        if no_grad:
            v.requires_grad = False
            freeze.append(k)
    print('fine_tune:', len(tune), tune)
    print('freeze', len(freeze), freeze)
    print('all', len(names))
    print('param', len(parameters))
    return parameters

if __name__ == "__main__":
    base_model = r2plus1d_34("r2plus1d_34_32_kinetics").cuda()
    print(base_model)
    input = torch.randn((8,3,32,112,112)).cuda()
    out = base_model.stem(input)
    out = base_model.layer1(out)
    out = base_model.layer2(out)
    print(out.size())
    # for i, b in enumerate(blocks):
    #     print(blocks[i].conv1[0][-1].in_channels)