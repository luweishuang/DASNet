import os
import torch
import torch.utils.data as Data
import utils.transforms as trans
import utils.utils as util
import layer.loss as ls

import shutil
import cfg.CDD as cfg
import dataset.rs as dates
import time
import datetime
from funcs import validate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resume = 0


def main():
    #########  configs ###########
    best_metric = 0
    ######  load datasets ########
    train_transform_det = trans.Compose([trans.Scale(cfg.TRANSFROM_SCALES),])
    val_transform_det = trans.Compose([trans.Scale(cfg.TRANSFROM_SCALES),])
    train_data = dates.Dataset(cfg.TRAIN_DATA_PATH, cfg.TRAIN_LABEL_PATH, cfg.TRAIN_TXT_PATH, 'train', transform=True, transform_med=train_transform_det)
    train_loader = Data.DataLoader(train_data, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_data = dates.Dataset(cfg.VAL_DATA_PATH, cfg.VAL_LABEL_PATH, cfg.VAL_TXT_PATH, 'val', transform=True, transform_med=val_transform_det)
    val_loader = Data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    ######  build  models ########
    base_seg_model = 'resnet50'
    if base_seg_model == 'vgg':
        import model.siameseNet.d_aa as models
        pretrain_deeplab_path = os.path.join(cfg.PRETRAIN_MODEL_PATH, 'vgg16.pth')
        model = models.SiameseNet(norm_flag='l2')
        if resume:
            checkpoint = torch.load(cfg.TRAINED_BEST_PERFORMANCE_CKPT)
            model.load_state_dict(checkpoint['state_dict'])
            print('resume success')
        else:
            deeplab_pretrain_model = torch.load(pretrain_deeplab_path)
            model.init_parameters_from_deeplab(deeplab_pretrain_model)
            print('load vgg')
    else:
        import model.siameseNet.dares as models
        model = models.SiameseNet(norm_flag='l2')
        if resume:
            checkpoint = torch.load(cfg.TRAINED_BEST_PERFORMANCE_CKPT)
            model.load_state_dict(checkpoint['state_dict'])
            print('resume success')
        else:
            print('load resnet50')
    if device.type == 'cuda':
        print("use cuda!")
        model = model.cuda()
    MaskLoss = ls.ContrastiveLoss1()
    ab_test_dir = os.path.join(cfg.SAVE_PRED_PATH, 'contrastive_loss')
    util.check_dir(ab_test_dir)
    save_change_map_dir = os.path.join(ab_test_dir, 'changemaps/')
    save_valid_dir = os.path.join(ab_test_dir,'valid_imgs')
    save_roc_dir = os.path.join(ab_test_dir,'roc')
    util.check_dir(save_change_map_dir),util.check_dir(save_valid_dir),util.check_dir(save_roc_dir)
    #########
    ######### optimizer ##########
    ######## how to set different learning rate for differernt layers #########
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.INIT_LEARNING_RATE, weight_decay=cfg.DECAY)
    ######## iter img_label pairs ###########
    loss_total = 0
    time_start = time.time()
    for epoch in range(60):
        for batch_idx, batch in enumerate(train_loader):
             step = epoch * len(train_loader) + batch_idx
             util.adjust_learning_rate(cfg.INIT_LEARNING_RATE, optimizer, step)
             model.train()
             img1, img2, label, filename, height, width = batch
             if device.type == 'cuda':
                 img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
             label = label.float()
             out_conv5, out_fc, out_embedding = model(img1, img2)
             out_conv5_t0, out_conv5_t1 = out_conv5
             out_fc_t0, out_fc_t1 = out_fc
             out_embedding_t0, out_embedding_t1 = out_embedding
             label_rz_conv5 = util.rz_label(label, size=out_conv5_t0.data.cpu().numpy().shape[2:])
             label_rz_fc = util.rz_label(label, size=out_fc_t0.data.cpu().numpy().shape[2:])
             label_rz_embedding = util.rz_label(label, size=out_embedding_t0.data.cpu().numpy().shape[2:])
             if device.type == 'cuda':
                 label_rz_conv5 = label_rz_conv5.cuda()
                 label_rz_fc = label_rz_fc.cuda()
                 label_rz_embedding = label_rz_embedding.cuda()
             contractive_loss_conv5 = MaskLoss(out_conv5_t0, out_conv5_t1, label_rz_conv5)
             contractive_loss_fc = MaskLoss(out_fc_t0, out_fc_t1, label_rz_fc)
             contractive_loss_embedding = MaskLoss(out_embedding_t0, out_embedding_t1, label_rz_embedding)
             loss = contractive_loss_conv5 + contractive_loss_fc + contractive_loss_embedding
             loss_total += loss.data.cpu()
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()
             if (batch_idx) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f Mask_Loss_conv5: %.4f Mask_Loss_fc: %.4f "
                      "Mask_Loss_embedding: %.4f" % (epoch, batch_idx,loss.item(),contractive_loss_conv5.item(),
                                                     contractive_loss_fc.item(),contractive_loss_embedding.item()))
             if (batch_idx) % 1000 == 0:
                 model.eval()
                 current_metric = validate(model, val_loader, epoch, save_change_map_dir, save_roc_dir, cfg.TRANSFROM_SCALES)
                 if current_metric > best_metric:
                     torch.save({'state_dict': model.state_dict()}, os.path.join(ab_test_dir, 'model' + str(epoch) + '.pth'))
                     shutil.copy(os.path.join(ab_test_dir, 'model' + str(epoch) + '.pth'), os.path.join(ab_test_dir, 'model_best.pth'))
                     best_metric = current_metric
        current_metric = validate(model, val_loader, epoch, save_change_map_dir, save_roc_dir, cfg.TRANSFROM_SCALES)
        if current_metric > best_metric:
            torch.save({'state_dict': model.state_dict()}, os.path.join(ab_test_dir, 'model' + str(epoch) + '.pth'))
            shutil.copy(os.path.join(ab_test_dir, 'model' + str(epoch) + '.pth'), os.path.join(ab_test_dir, 'model_best.pth'))
            best_metric = current_metric
        if epoch % 5 == 0:
            torch.save({'state_dict': model.state_dict()}, os.path.join(ab_test_dir, 'model' + str(epoch) + '.pth'))
    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False 
    main()
