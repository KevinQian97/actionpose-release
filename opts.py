import argparse
import time
import os
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('--dataset', type=str, default="DET")
parser.add_argument('--modality', type=str, choices=['RGB', 'Flow', "PoseAction"],default="RGB")
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
parser.add_argument('--out_path', type=str, default="./exp")
parser.add_argument('--call_path', type=str, default="/home/kevinq/repos/ActEV_Scorer")
parser.add_argument('--store_name', type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="R2plus1D")
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll',"bce","wbce"])
parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")
parser.add_argument('--suffix', type=str, default=None)
parser.add_argument('--pretrain', type=str, default='r2plus1d_34_32_kinetics')
parser.add_argument('--tune_from', type=str, default=None, help='fine-tune from checkpoint')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_scheduler_gamma', default=0.1, type=float, help='Learning rate decay factor')
parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--naudc', default=False, action="store_true", help='if calculate naudc')


# ========================= Runtime Configs ==========================

parser.add_argument('--vid_type', type=str, default=".avi")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_log',type=str, default='./logs')
parser.add_argument('--root_model', type=str, default='/models/')
parser.add_argument('--shuffle', default=False, action="store_true", help='use shuffle for neg sampling')
parser.add_argument('--trn_name', type=str, default="MEVA-kitware_eo_s2-train_158", help='training split name')
parser.add_argument('--tst_name', type=str, default="MEVA-kitware_eo_s2-test_99", help='testing split path')
parser.add_argument('--prop_path', type=str, default="", help='proposal path')
parser.add_argument('--label_path', type=str, default="", help='label path')
parser.add_argument('--cfg_file', type=str, default=None, help='slowfast based cfg files')
parser.add_argument("--neg_rate",type=float, default=0.5, help="rate of neg samples used for training")
parser.add_argument("--test_rate",type=float, default=0, help="rate of neg samples used for testing")
parser.add_argument("--topk",type=int, default=37, help="topk classes for evaluation")

# ========================= Models TSM&TSA ==========================
parser.add_argument('--shift', default=False, action="store_true", help='use shift for models')
parser.add_argument('--tsa', default=False, action="store_true", help='use tsa for models')
parser.add_argument('--stsa', default=False, action="store_true", help='use stsa for models')
parser.add_argument('--ttsa', default=False, action="store_true", help='use ttsa for models')
parser.add_argument('--me', default=False, action="store_true", help='use me for models')
parser.add_argument('--is_3D', default=False, action="store_true", help='if backbone contains temporal convolution')

parser.add_argument('--shift_div', default=8, type=int, help='number of div for shift (default: 8)')
parser.add_argument('--shift_place', default='blockres', type=str, help='place for shift (default: stageres)')
parser.add_argument('--shift_diff',  nargs='+', type=int, help='shift size parallel')
parser.add_argument('--shift_groups', default=2, type=int, help='number of shift groups')
parser.add_argument('--temporal_pool', default=False, action="store_true", help='add temporal pooling')
parser.add_argument('--non_local', default=False, action="store_true", help='add non local block')
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')


# ========================= Models VMZ ==========================
parser.add_argument('--vmz_tune_last_k_layer', default=1, type=int,
                    help='Fine tune last k layers, range is [0,4]')
parser.add_argument('--all_sample', default=False, action="store_true", help='use all sample for MEVA')
parser.add_argument("--optimizer", default="adam", type=str, help="optimizer types")
parser.add_argument("--lr_scheduler",default=False, action="store_true",help="whether use scheduler")
