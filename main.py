import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import logging
import sys
import csv
import json
from torch.nn.utils import clip_grad_norm_
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy_nll, accuracy_bce, get_eventdict, getoutput
from ops.TSA import make_temporal_pool
from archs.R2plus1D import get_vmz_fine_tuning_parameters
from tensorboardX import SummaryWriter
from tools.gen_label_strkf2 import gen_label
from ops.loss import WeightedBCEWithLogitsLoss,prep_weight
best_prec1 = 1
least_loss = 1000


def main():
    global args, best_prec1, least_loss
    least_loss = 1000
    args = parser.parse_args()
    if os.path.exists(os.path.join(args.root_log,"error.log")):
        os.remove(os.path.join(args.root_log,"error.log"))
    logging.basicConfig(level=logging.DEBUG,filename=os.path.join(args.root_log,"error.log"),
        filemode='a',
        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    # log_handler = open(os.path.join(args.root_log,"error.log"),"w")
    # sys.stdout = log_handler
    if args.root_path:
        num_class, args.train_list, args.val_list, _, prefix = dataset_config.return_dataset(args.dataset, args.modality)
        args.train_list = os.path.join(args.root_log,"kf1_train_anno_lijun_iod.json")
        args.test_list = os.path.join(args.root_log,"kf1_test_anno_lijun_iod.json")
    else:
        num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset, args.modality)

    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['TSA', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    # if args.pretrain != 'imagenet':
    #     args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)

    check_rootfolders()

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local,
                is_TSA=args.tsa,
                is_sTSA=args.stsa,
                is_tTSA = args.ttsa,
                shift_diff=args.shift_diff,
                shift_groups=args.shift_groups,
                is_ME = args.me,
                is_3D = args.is_3D,
                cfg_file=args.cfg_file)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    # policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    if args.optimizer=="sgd":
        if args.lr_scheduler:
            optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optimizer=="adam":
        params = get_vmz_fine_tuning_parameters(model, args.vmz_tune_last_k_layer)
        optimizer = torch.optim.Adam(params, 
                                    args.lr, 
                                    weight_decay=args.weight_decay)
    else:
        raise RuntimeError("not supported optimizer")

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.lr_scheduler_gamma)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # if args.lr_scheduler:
            #     scheduler.load_state_dict(checkpoint["lr_scheduler"])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
            logging.info(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
            logging.error(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        # sd = {k:v for k, v in sd.items() if k in keys2}
        sd = {k: v for k, v in sd.items() if k in keys2}
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k and "projection" not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality in ['RGB', "PoseAction"]:
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5
    if not args.shuffle:
        train_loader = torch.utils.data.DataLoader(
            TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3']),inc_dim=(args.arch in ["R2plus1D","X3D"])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3']),inc_dim=(args.arch in ["R2plus1D","X3D"])),
                       normalize,
                   ]), dense_sample=args.dense_sample, all_sample=args.all_sample),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
            drop_last=True)  # prevent something not % n_GPU

        val_loader = torch.utils.data.DataLoader(
            TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(scale_size),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3']),inc_dim=(args.arch in ["R2plus1D","X3D"])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3']),inc_dim=(args.arch in ["R2plus1D","X3D"])),
                       normalize,
                   ]), dense_sample=args.dense_sample, all_sample=args.all_sample),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    

    # for group in policies:
    #     print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
    #         group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.evaluate:
        if args.loss_type == 'nll':
            criterion = torch.nn.CrossEntropyLoss().cuda()
        elif args.loss_type == "bce":
            criterion = torch.nn.BCEWithLogitsLoss().cuda()
        elif args.loss_type == "wbce": 
            class_weight,pos_weight = prep_weight(args.train_list)
            criterion = WeightedBCEWithLogitsLoss(class_weight,pos_weight)
        else:
            raise ValueError("Unknown loss type")

        val_loader = torch.utils.data.DataLoader(
            TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(scale_size),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3']),inc_dim=(args.arch in ["R2plus1D","X3D"])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3']),inc_dim=(args.arch in ["R2plus1D","X3D"])),
                       normalize,
                   ]), dense_sample=args.dense_sample, all_sample=args.all_sample,analysis=True),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        test(val_loader, model, criterion, 0)
        return

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    print(model)
    logging.info(model)
    for epoch in range(args.start_epoch, args.epochs):
        logging.info("Train Epoch {}/{} starts, estimated time 5832s".format(str(epoch),str(args.epochs)))
        # update data_loader
        if args.shuffle:
            gen_label(args.prop_path,args.label_path,args.trn_name,args.train_list,args.neg_rate,STR=False)
            gen_label(args.prop_path,args.label_path,args.tst_name,args.val_list,args.test_rate,STR=False)
            train_loader = torch.utils.data.DataLoader(
            TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3']),inc_dim=(args.arch in ["R2plus1D","X3D"])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3']),inc_dim=(args.arch in ["R2plus1D","X3D"])),
                       normalize,
                   ]), dense_sample=args.dense_sample, all_sample=args.all_sample),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
            drop_last=True) 

            val_loader = torch.utils.data.DataLoader(
            TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(scale_size),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3']),inc_dim=(args.arch in ["R2plus1D","X3D"])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3']),inc_dim=(args.arch in ["R2plus1D","X3D"])),
                       normalize,
                   ]), dense_sample=args.dense_sample, all_sample=args.all_sample),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
            print(train_loader)

            # define loss function (criterion) and optimizer
        if args.loss_type == 'nll':
            criterion = torch.nn.CrossEntropyLoss().cuda()
        elif args.loss_type == "bce":
            criterion = torch.nn.BCEWithLogitsLoss().cuda()
        elif args.loss_type == "wbce": 
            class_weight,pos_weight = prep_weight(args.train_list)
            criterion = WeightedBCEWithLogitsLoss(class_weight,pos_weight)
        else:
            raise ValueError("Unknown loss type")

        if not args.lr_scheduler:
            adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)
            train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)
            
        else:
            train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)
            scheduler.step()
            
            
        # train for one epoch
        

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            logging.info("Test Epoch {}/{} starts, estimated time 13874s".format(str(epoch // args.eval_freq),str(args.epochs/args.eval_freq)))
            if args.loss_type == "wbce": 
                # class_weight,pos_weight = prep_weight(args.val_list)
                criterion = torch.nn.BCEWithLogitsLoss().cuda()
            lossm = validate(val_loader, model, criterion, epoch, log_training, tf_writer)

            # remember best prec@1 and save checkpoint
            is_best = lossm < least_loss
            least_loss = min(lossm, least_loss)
            tf_writer.add_scalar('lss/test_top1_best', least_loss, epoch)

            output_best = 'Best Loss: %.3f\n' % (lossm)
            logging.info(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()
            if args.lr_scheduler:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': least_loss,
                    'lr_scheduler': scheduler,
                }, is_best,epoch)
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': least_loss,
                }, is_best,epoch)

def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    num_data = len(train_loader)
    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        # input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input)
        optimizer.zero_grad()
        # loss = criterion(output, target_var)
        loss = criterion(output.squeeze(-1), target_var)

        # measure accuracy and record loss
        if args.loss_type == "bce" or args.loss_type == "wbce":
            [prec1, prec5], num = accuracy_bce(output.data, target, topk=(1, 5))
            losses.update(loss.item(), output.size(0))
            if prec1 != -1:
                top1.update(prec1.item(), num)
                top5.update(prec5.item(), num)
        elif args.loss_type == "nll":
            [prec1, prec5] = accuracy_nll(output.data, target, topk=(1,5))
            losses.update(loss.item(), output.size(0))
            top1.update(prec1.item(), output.size(0))
            top5.update(prec5.item(), output.size(0))

        # compute gradient and do SGD step
        

        loss.backward()

        # if args.clip_gradient is not None:
        #     total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))  # TODO
            print(output)
            logging.info(output)
            log.write(output + '\n')
            log.flush()
        tf_writer.add_scalar("loss/train",losses.val, i+epoch*num_data)
        tf_writer.add_scalar('acc/train_top1', top1.val, i+epoch*num_data)
        tf_writer.add_scalar('acc/train_top5', top5.val, i+epoch*num_data)
        tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    nAUDC = 1
    pMiss = 1

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        if args.naudc:
            naudc_res = []
        for i, (input, target) in enumerate(val_loader):
            batch_size = target.numel()
            target = target.cuda()

            # compute output
            output = model(input)
            # if args.dense_sample:
            #     output = output.reshape(batch_size, 10, -1).mean(1)
            # loss = criterion(output, target)
            loss = criterion(output.squeeze(-1), target)
            # measure accuracy and record loss
            if args.loss_type == "bce" or args.loss_type == "wbce":
                [prec1, prec5], num = accuracy_bce(output.data, target, topk=(1, 5))
                losses.update(loss.item(), output.size(0))
                if prec1 != -1:
                    top1.update(prec1.item(), num)
                    top5.update(prec5.item(), num)
            elif args.loss_type == "nll":
                [prec1, prec5] = accuracy_nll(output.data, target, topk=(1,5))
                losses.update(loss.item(), output.size(0))
                top1.update(prec1.item(), output.size(0))
                top5.update(prec5.item(), output.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.naudc:
                rst = output.data.cpu().numpy().copy()
                this_rst_list = []
                this_rst_list.append(rst)
                ensembled_predict = sum(this_rst_list) / len(this_rst_list)
                for p, g in zip(ensembled_predict, target.cpu().numpy()):
                    naudc_res.append([p[None, ...], g])
            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                logging.info(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()
    if args.naudc:
        video_pred_topall = [np.argsort(np.mean(x[0], axis=0).reshape(-1))[::-1][:args.topk] for x in naudc_res]
        video_prob_topall = [np.sort(np.mean(x[0], axis=0).reshape(-1))[::-1][:args.topk] for x in naudc_res]
        with open(args.val_list) as f:
            vid_names = list(json.load(f)["database"].keys())
        event_dict = get_eventdict(args)
        res_dict,file_dict,eve_dict = getoutput(vid_names,video_pred_topall,video_prob_topall,event_dict,args)
        
        json_str = json.dumps(res_dict,indent=4)
        with open(os.path.join(args.out_path,"output-mod.json"), 'w') as save_json:
            save_json.write(json_str) 
        
        json_str = json.dumps(file_dict,indent=4)
        with open(os.path.join(args.out_path,"file-index.json"), 'w') as save_json:
            save_json.write(json_str)

        json_str = json.dumps(eve_dict,indent=4)
        with open(os.path.join(args.out_path,"activity-index.json"), 'w') as save_json:
            save_json.write(json_str)
        
        call_path = os.path.join(args.call_path,"ActEV_Scorer.py")
        s = os.path.join(args.out_path,"output_mod.json")
        r = os.path.join(args.out_path,"kitware_eo_s2-test_99.json")
        a = os.path.join(args.out_path,"activity-index.json")
        f = os.path.join(args.out_path,"file-index.json")
        o = args.out_path
        print("Start Calculatomg Pmiss and nAUDC")
        call = "python {} ActEV_SDL_V2 -s {} -r {} -a {} -f {} -o {} -v -n 36".format(call_path,s,r,a,f,o)
        os.system(call)
        csvFile = open(os.path.join(args.out_path,"scores_aggregated.csv"),"r")
        reader = csv.reader(csvFile)
        for item in reader:
            if "mean-nAUDC@0.2tfa" in item[0]:
                nAUDC = float(item[0].split("|")[-1])
            if "mean-p_miss@0.02tfa" in item[0]:
                pMiss = float(item[0].split("|")[-1])


    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f} nAUDC {nAUDC:.5f} pMiss {pMiss:.5f}'
              .format(top1=top1, top5=top5, loss=losses, nAUDC=nAUDC, pMiss=pMiss))
    print(output)
    logging.info(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)
        tf_writer.add_scalar('nAUDC/test', nAUDC, epoch)
        tf_writer.add_scalar('pmiss/test', pMiss, epoch)

    return losses.avg

def test(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    nAUDC = 1
    pMiss = 1
    pred_dict = {}
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        if args.naudc:
            naudc_res = []
        for i, (input, target, name) in enumerate(val_loader):
            batch_size = target.numel()
            target = target.cuda()

            # compute output
            output = model(input)
            # if args.dense_sample:
            #     output = output.reshape(batch_size, 10, -1).mean(1)
            # loss = criterion(output, target)
            loss = criterion(output.squeeze(-1), target)
            # measure accuracy and record loss
            if args.loss_type == "bce" or args.loss_type == "wbce":
                [prec1, prec5], num = accuracy_bce(output.data, target, topk=(1, 5))
                losses.update(loss.item(), output.size(0))
                if prec1 != -1:
                    top1.update(prec1.item(), num)
                    top5.update(prec5.item(), num)
            elif args.loss_type == "nll":
                [prec1, prec5] = accuracy_nll(output.data, target, topk=(1,5))
                losses.update(loss.item(), output.size(0))
                top1.update(prec1.item(), output.size(0))
                top5.update(prec5.item(), output.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            rst = torch.softmax(output,dim=1).data.cpu().numpy().copy()
            tgt = target.cpu().numpy().copy()

            for idk in range(rst.shape[0]):
                pred_dict[str(name[idk])] = {"pred":rst[idk].tolist(),"gt":tgt[idk].tolist()}
                

            if args.naudc:
                rst = output.data.cpu().numpy().copy()
                this_rst_list = []
                this_rst_list.append(rst)
                ensembled_predict = sum(this_rst_list) / len(this_rst_list)
                for p, g in zip(ensembled_predict, target.cpu().numpy()):
                    naudc_res.append([p[None, ...], g])
            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                logging.info(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    json_str = json.dumps(pred_dict,indent=4)
    with open(os.path.join("./pred.json"), 'w') as save_json:
        save_json.write(json_str) 

    if args.naudc:
        video_pred_topall = [np.argsort(np.mean(x[0], axis=0).reshape(-1))[::-1][:args.topk] for x in naudc_res]
        video_prob_topall = [np.sort(np.mean(x[0], axis=0).reshape(-1))[::-1][:args.topk] for x in naudc_res]
        with open(args.val_list) as f:
            vid_names = list(json.load(f)["database"].keys())
        event_dict = get_eventdict(args)
        res_dict,file_dict,eve_dict = getoutput(vid_names,video_pred_topall,video_prob_topall,event_dict,args)
        
        json_str = json.dumps(res_dict,indent=4)
        with open(os.path.join(args.out_path,"output-mod.json"), 'w') as save_json:
            save_json.write(json_str) 
        
        json_str = json.dumps(file_dict,indent=4)
        with open(os.path.join(args.out_path,"file-index.json"), 'w') as save_json:
            save_json.write(json_str)

        json_str = json.dumps(eve_dict,indent=4)
        with open(os.path.join(args.out_path,"activity-index.json"), 'w') as save_json:
            save_json.write(json_str)
        
        call_path = os.path.join(args.call_path,"ActEV_Scorer.py")
        s = os.path.join(args.out_path,"output_mod.json")
        r = os.path.join(args.out_path,"kitware_eo_s2-test_99.json")
        a = os.path.join(args.out_path,"activity-index.json")
        f = os.path.join(args.out_path,"file-index.json")
        o = args.out_path
        print("Start Calculatomg Pmiss and nAUDC")
        call = "python {} ActEV_SDL_V2 -s {} -r {} -a {} -f {} -o {} -v -n 36".format(call_path,s,r,a,f,o)
        os.system(call)
        csvFile = open(os.path.join(args.out_path,"scores_aggregated.csv"),"r")
        reader = csv.reader(csvFile)
        for item in reader:
            if "mean-nAUDC@0.2tfa" in item[0]:
                nAUDC = float(item[0].split("|")[-1])
            if "mean-p_miss@0.02tfa" in item[0]:
                pMiss = float(item[0].split("|")[-1])


    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f} nAUDC {nAUDC:.5f} pMiss {pMiss:.5f}'
              .format(top1=top1, top5=top5, loss=losses, nAUDC=nAUDC, pMiss=pMiss))
    print(output)
    logging.info(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)
        tf_writer.add_scalar('nAUDC/test', nAUDC, epoch)
        tf_writer.add_scalar('pmiss/test', pMiss, epoch)

    return losses.avg


def save_checkpoint(state, is_best,epoch):
    if is_best:
        filename = '%s/%s/best.pth.tar' % (args.root_model, args.store_name)
        torch.save(state, filename)
    filename = '%s/%s/%s.pth.tar' % (args.root_model, args.store_name,epoch)
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
