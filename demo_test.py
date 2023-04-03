import cv2
import torch
from demo_dataset import IrisDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from demo_dataset import transform
import os
from demo_opt import parse_args
from models import model_dict
from tqdm import tqdm
from utils import get_predictions


def mkdirs():
    os.makedirs('test/labels/', exist_ok=True)
    os.makedirs('test/output/', exist_ok=True)
    os.makedirs('test/mask/', exist_ok=True)
    os.makedirs('Semantic_Segmentation_Dataset/test/output/', exist_ok=True)


def main():
    device = torch.device("cuda")

    model = model_dict['densenet']
    model = model.to(device)
    filename = 'best_model.pkl'
    if not os.path.exists(filename):
        print("model path not found !!!")
        exit(1)

    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()

    test_set = IrisDataset(filepath='Semantic_Segmentation_Dataset/', split='test', transform=transform)
    testloader = DataLoader(test_set, batch_size=2, shuffle=False, num_workers=2)

    mkdirs()

    with torch.no_grad():
        for i, batchdata in tqdm(enumerate(testloader), total=len(testloader)):
            img, labels, index, x, y = batchdata
            data = img.to(device)
            output = model(data)
            predict = get_predictions(output)
            for j in range(len(index)):
                np.save('test/labels/{}.npy'.format(index[j]), predict[j].cpu().numpy())
                try:
                    plt.imsave('test/output/{}.jpg'.format(index[j]), 255 * labels[j].cpu().numpy())
                except:
                    pass

                pred_img = predict[j].cpu().numpy() / 3.0

                pred_img = cv2.normalize(src=pred_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # pred_img = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
                # pred_img = cv2.applyColorMap(pred_img, 13)
                pred_img = cv2.applyColorMap(pred_img, 16)

                # inp = img[j].squeeze() * 0.5 + 0.5
                # img_orig = np.clip(inp, 0, 1)
                # img_orig = np.array(img_orig)
                # combine = np.hstack([img_orig, pred_img])

                # pred_img = cv2.convertScaleAbs(pred_img, alpha=(255.0))
                # pred_img = cv2.normalize(src=pred_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # pred_img = cv2.normalize(pred_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # from skimage import img_as_ubyte
                #
                # pred_img = img_as_ubyte(pred_img)
                cv2.imwrite('Semantic_Segmentation_Dataset/test/output/{}.jpg'.format(index[j]), pred_img)

                # plt.imsave('Semantic_Segmentation_Dataset/test/output/{}.jpg'.format(index[j]), pred_img)

    # os.rename('test', args.save)


if __name__ == '__main__':
    main()
