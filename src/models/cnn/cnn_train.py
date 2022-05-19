from fastai.vision.all import *
import torch


def label_func(file):
    status = file.split('-')[1][0]
    code = file[-8:-4]
    if code.count('0') == 4:
        syn = 0
    else:
        count = (
            (1, code.count('1')),
            (2, code.count('2')),
            (3, code.count('3')),
            (4, code.count('4')),
            (5, code.count('5'))
        )
        syn = max(count, key=itemgetter(1))[0]
    code = status + str(syn)
    covid = ['33', '03', '30', '23', '31', '32', '34']
    other = ['15', '05', '10', '25', '21', '22', '24', '01', '02', '04', '20']
    if code in covid:
        return True
    elif code in other:
        return False


def train_model(
    data_path,
    export_path,
    model=models.resnet18,
    epochs=1,
    metrics=accuracy,
    loss_func=None,
    optimizer=Adam,
    lr=0.001
    ):
    data_path, export_path = Path(data_path), Path(export_path)
    # dataloader
    dls = ImageDataLoaders.from_name_func(
        path=data_path,
        fnames=get_image_files(data_path),
        label_func=label_func,
        item_tfms=Resize(1025, pad_mode='zeros'),
        valid_pct=0.2,
        seed=3,
        bs=4,
        num_workers=0,
        device=torch.device('cuda')
    )
    # learner
    learn = vision_learner(
        dls=dls,
        arch=model,
        metrics=metrics,
        loss_func=loss_func,
        opt_func=optimizer,
        lr=lr,
    )
    # train
    learn.fine_tune(epochs)
    # export
    learn.export(export_path)