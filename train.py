#-------------------------------------#
#       HUẤN LUYỆN MÔ HÌNH (TRAIN)
#-------------------------------------#
import datetime
import os
from functools import partial
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (YOLOLoss, get_lr_scheduler, set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (get_classes, seed_everything, show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    # ====================================================================
    #                           PHẦN 1: CẤU HÌNH
    # ====================================================================

    Cuda            = True

    classes_path    = 'model_data/my_classes.txt'
    model_path      = 'model_data/yolov4_tiny_weights_coco.pth'
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_train.txt'

    input_shape     = [416, 416]
    anchors_mask    = [[3, 4, 5], [0, 1, 2]] 
    
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 32  
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 16 
    
    Freeze_Train        = True 

    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4

    save_period         = 10   
    save_dir            = 'logs' 
    num_workers         = 4   

    pretrained          = False
    eval_flag           = True
    eval_period         = 10
    distributed         = False
    sync_bn             = False
    fp16                = False
    label_smoothing     = 0
    mosaic              = False
    mosaic_prob         = 0.5
    mixup               = False
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    lr_decay_type       = "cos"
    phi                 = 0

    # ====================================================================
    #                       PHẦN 2: KHỞI TẠO HỆ THỐNG
    # ====================================================================
    seed_everything(11)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0
    
    class_names, num_classes = get_classes(classes_path)
    
    anchors = np.array([
        [10, 14],  [23, 27],   [37, 58], 
        [81, 82],  [135, 169], [344, 319]
    ], dtype='float32')
    num_anchors = 6


    model = YoloBody(anchors_mask, num_classes, pretrained=pretrained, phi=phi)
    
    if not pretrained:
        weights_init(model)
        
    if model_path != '':
        print(f'Load weights {model_path}.')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        print("--> Load weights thành công!")

    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing)
    
    time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # Đọc dữ liệu
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train = len(train_lines)
    num_val   = len(val_lines)

    show_config(
        classes_path=classes_path, anchors_path='Hardcoded', anchors_mask=anchors_mask, model_path=model_path, input_shape=input_shape, \
        Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch, Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type, \
        save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
    )

    # ====================================================================
    #                       PHẦN 3: VÒNG LẶP HUẤN LUYỆN
    # ====================================================================
    if True:
        UnFreeze_flag = False
        
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit  = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        train_dataset   = YoloDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        
        gen     = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=0, seed=11))
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=0, seed=11))

        eval_callback = EvalCallback(model, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines, log_dir, Cuda, eval_flag=eval_flag, period=eval_period)

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit  = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset quá nhỏ, hãy tăng số lượng ảnh!")

                gen     = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=0, seed=11))
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=0, seed=11))

                UnFreeze_flag = True

            gen.dataset.epoch_now       = epoch
            gen_val.dataset.epoch_now   = epoch
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, 
                          num_train // batch_size, num_val // batch_size, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, None, save_period, save_dir, local_rank)


        loss_history.writer.close()
