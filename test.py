import os
import torch
import torch.utils.data as Data
import utils.transforms as trans
import time
import datetime
from funcs import validate, test

datasets = 'IEEMOO'
if datasets == 'CDD':
    import cfg.CDD as cfg
    import dataset.CDD as dates
if datasets == 'IEEMOO':
    import cfg.IEEMOO as cfg
    import dataset.IEEMOO as dates
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main_valid():
    val_transform_det = trans.Compose([trans.Scale(cfg.TRANSFROM_SCALES),])
    BASE_PATH = os.path.join(os.path.abspath(os.getcwd()), "use")
    val_data = dates.Dataset(os.path.join(BASE_PATH, 'changedetection/SceneChangeDet/CDD'), os.path.join(BASE_PATH, 'changedetection/SceneChangeDet/CDD'),
                             os.path.join(BASE_PATH, 'changedetection/SceneChangeDet/CDD/val_small.txt'), 'val',
                             transform=True, transform_med=val_transform_det)
    val_loader = Data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # import model.siameseNet.dares as models
    import model.siameseNet.d_aa as models
    model = models.SiameseNet(norm_flag='l2')
    checkpoint = torch.load(os.path.join(BASE_PATH, 'cdout/bone/resnet50/mcanshu/ckpt/CDD_model_best.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    print('load success')
    if device.type == 'cuda':
        model = model.cuda()
    save_change_map_dir = os.path.join(cfg.SAVE_PRED_PATH, 'contrastive_loss/changemaps/')
    save_roc_dir = os.path.join(cfg.SAVE_PRED_PATH, 'contrastive_loss/roc')
    time_start = time.time()
    current_metric = validate(model, val_loader, 100, save_change_map_dir, save_roc_dir, cfg.TRANSFROM_SCALES)
    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))


def main_test():
    test_transform_det = trans.Compose([trans.Scale(cfg.TRANSFROM_SCALES),])
    BASE_PATH = os.path.join(os.path.abspath(os.getcwd()), "use")
    test_data = dates.Dataset(os.path.join(BASE_PATH, 'changedetection/SceneChangeDet/back_42fps'), os.path.join(BASE_PATH, 'changedetection/SceneChangeDet/back_42fps'),
                             os.path.join(BASE_PATH, 'changedetection/SceneChangeDet/back_42fps/test.txt'), 'test',
                             transform=True, transform_med=test_transform_det)
    test_loader = Data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # import model.siameseNet.dares as models
    import model.siameseNet.d_aa as models
    model = models.SiameseNet(norm_flag='l2')
    checkpoint = torch.load(os.path.join(BASE_PATH, 'cdout/bone/resnet50/mcanshu/ckpt/CDD_model_best.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    print('load success')
    if device.type == 'cuda':
        model = model.cuda()
    save_change_map_dir = os.path.join(cfg.SAVE_PRED_PATH, 'contrastive_loss/changemaps/')
    save_roc_dir = os.path.join(cfg.SAVE_PRED_PATH, 'contrastive_loss/roc')
    time_start = time.time()
    current_metric = test(model, test_loader, 100, save_change_map_dir, save_roc_dir, cfg.TRANSFROM_SCALES)
    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    # main_valid()
    main_test()

