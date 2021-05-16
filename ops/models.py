from torch import nn

from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_
import os
import copy
from ops.st_gcn import ST_GCN_18

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False,shift_groups=2,shift_diff=[3,5],
                 is_TSA = False,is_sTSA = False, is_ME = False, is_3D = False, is_tTSA = False,
                 cfg_file = None):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local
        self.shift_groups = shift_groups
        self.shift_diff = shift_diff
        self.is_TSA = is_TSA
        self.is_sTSA = is_sTSA
        self.is_ME = is_ME
        self.is_3D = is_3D
        self.is_tTSA = is_tTSA
        self.cfg_file = cfg_file
        

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            if modality == "RGB":
                self.new_length = 1 
            elif modality=="TwoStream":
                self.new_length = 1 
            else:
                self.new_length = 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)
        feat_pos = 256

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")
        elif self.modality == "TwoStream":
            print("Converting the ImageNet model to TwoStream model")
            self.stream_model = copy.deepcopy(self.base_model)
        elif self.modality == "PoseAction":
            self.pose_model = ST_GCN_18(in_channels=3, num_class=60, graph_cfg={'layout': 'ntu-rgb+d', 'strategy': 'spatial'}, dropout=0.5)
            print("Adding Pose Module")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)
    def _construct_pose_model(self, num_class = 60):
        return ST_GCN_18(num_class=num_class)

    def _prepare_tsn(self, num_class):
        if self.base_model_name == "X3D":
            feature_dim = getattr(self.base_model.head, self.base_model.last_layer_name).in_features
        else:
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            if self.base_model_name == "X3D":
                setattr(self.base_model.head, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            else:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            if self.base_model_name == "X3D":
                setattr(self.base_model.head, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            else:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            if self.modality == "TwoStream":
                self.new_fc = nn.Linear(feature_dim*2, num_class)
            elif self.modality == "PoseAction":
                self.proj =nn.Linear(256,feature_dim)
                self.pose_relu = nn.ReLU()
                self.pose_drop = nn.Dropout(0.5)
                self.new_fc = nn.Linear(feature_dim*2, num_class)
            else:
                self.new_fc = nn.Linear(feature_dim,num_class)

        std = 0.001
        if self.new_fc is None:
            if self.base_model_name == "X3D":
                normal_(getattr(self.base_model.head, self.base_model.last_layer_name).weight, 0, std)
                constant_(getattr(self.base_model.head, self.base_model.last_layer_name).bias, 0)
            else:
                normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
                constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)
            if self.is_sTSA:
                print('Adding spatial aware temporal selective aggregation...')
                from ops.sTSA import make_sTSA
                make_sTSA(self.base_model, self.num_segments,
                                    n_div=self.shift_div, temporal_pool=self.temporal_pool)
            if self.is_TSA:
                print("Adding temporal selective aggregation...")
                from ops.TSA import make_TSA
                make_TSA(self.base_model, self.num_segments,
                                    n_div=self.shift_div, temporal_pool=self.temporal_pool,shift_groups=self.shift_groups, shift_diffs=self.shift_diff)
            if self.is_shift:
                print('Adding temporal shift...')
                from ops.temporal_shift import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, temporal_pool=self.temporal_pool)
            
            if self.is_ME:
                print("Adding temporal ME...")
                from ops.ME import make_temporal_me
                make_temporal_me(self.base_model)
            
            if self.is_tTSA:
                print("Adding tTSA module...")
                from ops.tTSA import make_tTSA
                make_tTSA(self.base_model, self.num_segments,
                                    n_div=self.shift_div, temporal_pool=self.temporal_pool)
            
            if self.non_local:
                print('Adding non-local module...')
                from ops.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'mobilenetv2':
            from archs.mobilenet_v2 import mobilenet_v2, InvertedResidual
            self.base_model = mobilenet_v2(True if self.pretrain == 'imagenet' else False)

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.is_shift:
                from ops.adaptive_temporal_shift import TemporalShift
                for m in self.base_model.modules():
                    if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                        if self.print_spec:
                            print('Adding temporal shift... {}'.format(m.use_res_connect))
                        m.conv[0] = TemporalShift(m.conv[0], n_segment=self.num_segments, n_div=self.shift_div)
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'BNInception':
            from archs.bn_inception import bninception
            self.base_model = bninception(pretrained=self.pretrain)
            self.input_size = self.base_model.input_size
            self.input_mean = self.base_model.mean
            self.input_std = self.base_model.std
            self.base_model.last_layer_name = 'fc'
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
            if self.is_shift:
                print('Adding temporal shift...')
                self.base_model.build_temporal_ops(
                    self.num_segments, is_temporal_shift=self.shift_place, shift_div=self.shift_div)
                    
        elif base_model == "R2plus1D":
            from archs.R2plus1D import r2plus1d_34
            self.base_model = r2plus1d_34("r2plus1d_34_32_kinetics")

            if self.is_ME:
                print("Adding temporal ME...")
                from ops.ME import make_temporal_me
                make_temporal_me(self.base_model)


            self.input_size = 112
            self.input_mean =[0.43216, 0.394666, 0.37645]
            self.input_std = [0.22803, 0.22145, 0.216989]
            self.base_model.last_layer_name = 'fc'
        

        elif base_model == "X3D":
            from archs.X3D import X3D,build_model
            self.base_model = build_model(self.cfg_file)
            checkpoint = torch.load("/data/yijunq/models/TSA/x3d_l.pyth")
            self.base_model.load_state_dict(checkpoint["model_state"])
            self.input_size = 312
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            self.base_model.last_layer_name = 'projection'
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
            if self.modality == "TwoStream":
                print("Freezing BatchNorm2D except the first one for stream branch.")
                for m in self.stream_model.modules():
                    if isinstance(m,nn.BatchNorm2d):
                        count+=1
                        if count >= (2 if self._enable_pbn else 1):
                            m.eval()
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False


    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]


    def forward(self, input, no_reshape=False):
        if self.modality == "PoseAction":
            vid = input["vid"]
            joints = input["joints"]
            base_out = self.base_model(vid)
            joint_out = self.pose_model(joints)
            joint_out = self.proj(joint_out)
            joint_out = self.pose_relu(joint_out)
            joint_out = self.pose_drop(joint_out)
            feat = torch.cat((base_out,joint_out),dim=1)
            return self.new_fc(feat)

        elif self.is_3D:
            if not no_reshape:
                sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

                if self.modality == 'RGBDiff':
                    sample_len = 3 * self.new_length
                    input = self._get_diff(input)
                # print(input.size())
                # input = input.view((-1,self.num_segments,3)+input.size()[-2:])
                # input = input.transpose(1,2)
                # raise RuntimeError("stop")
                if self.base_model_name == "X3D":
                    # print(input.size())
                    base_out = self.base_model([input])
                else:
                    base_out = self.base_model(input) 
            else:
                base_out = self.base_model(input)

            if self.dropout > 0:
                base_out = self.new_fc(base_out)

            if not self.before_softmax:
                base_out = self.softmax(base_out,dim=1)
            return base_out
        
        else:
            if not no_reshape:
                sample_len = (3 if self.modality == "RGB" or self.modality == "TwoStream" else 2) * self.new_length

                if self.modality == 'RGBDiff':
                    sample_len = 3 * self.new_length
                    input = self._get_diff(input)
                if self.modality == "TwoStream":
                    base_input = input[:,0,:,:,:]
                    # print(base_input.size())
                    # raise RuntimeError("stop")
                    stream_input = input[:,1,:,:,:]

                if self.modality == "TwoStream":
                    base_out = self.base_model(base_input.contiguous().view((-1, sample_len)+input.size()[-2:]))
                    stream_out = self.stream_model(stream_input.contiguous().view((-1, sample_len)+input.size()[-2:]))
                    base_out = torch.cat([base_out,stream_out],dim = 1)
                    
                else:
                    base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
            else:
                base_out = self.base_model(input)

            if self.dropout > 0:
                base_out = self.new_fc(base_out)

            if not self.before_softmax:
                base_out = self.softmax(base_out)
            # print(base_out.size())
            if self.reshape:
                if self.is_shift and self.temporal_pool:
                    base_out = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])
                else:
                    base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
                # print(base_out.size())
                output = self.consensus(base_out)
            return output.squeeze(1)



    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        if self.base_model_name == 'BNInception':
            import torch.utils.model_zoo as model_zoo
            sd = model_zoo.load_url('https://www.dropbox.com/s/35ftw2t4mxxgjae/BNInceptionFlow-ef652051.pth.tar?dl=1')
            base_model.load_state_dict(sd)
            print('=> Loading pretrained Flow weight done...')
        else:
            print('#' * 30, 'Warning! No Flow pretrained model is found')
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return [self.input_size,self.input_size]

    @property
    def scale_size(self):
        if self.is_3D:
            if self.base_model_name == "X3D":
                return [356, 446]
            elif self.base_model_name == "R2plus1D":
                return [128,171]
            else:
                raise RuntimeError("not inmplemented base model:{}".format(self.base_model_name))
        else:
            return self.input_size * 256 // self.input_size


    def get_augmentation(self, flip=True):
        if self.modality in ["RGB", "PoseAction"]:
            if self.is_3D:
                if flip:
                    print("Simple center crop")
                    return torchvision.transforms.Compose([GroupScale(self.scale_size),GroupMultiScaleCrop(self.input_size, [1, .875, 0.75,0.66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
                else:
                    print("Simple center crop without flip")
                    return torchvision.transforms.Compose([GroupScale(self.scale_size),GroupMultiScaleCrop(self.input_size, [1, .875, 0.75])])
            
            elif flip:
                return torchvision.transforms.Compose([GroupScale(self.scale_size),GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            else:
                print('#' * 20, 'NO FLIP!!!')
                return torchvision.transforms.Compose([GroupScale(self.scale_size),GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
        elif self.modality == "TwoStream":
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                        GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
