import os
import torch
import torch.utils.data as Data
import utils.transforms as trans
import time
import datetime
import dataset.rs as dates
import cfg.CDD as cfg
from funcs import validate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    val_transform_det = trans.Compose([
        trans.Scale(256, 256),
    ])
    BASE_PATH = os.path.join(os.path.abspath(os.getcwd()), "use")
    val_data = dates.Dataset(os.path.join(BASE_PATH, 'changedetection/SceneChangeDet/CDD'), os.path.join(BASE_PATH, 'changedetection/SceneChangeDet/CDD'),
                             os.path.join(BASE_PATH, 'changedetection/SceneChangeDet/CDD/val.txt'), 'val',
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
    current_metric = validate(model, val_loader, 1, save_change_map_dir, save_roc_dir)
    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))


if __name__ == '__main__':
   main()

