import torch
import trash_can_coco_torch
import torchvision
from torchvision import models
from engine import train_one_epoch, evaluate
import utils
import torchvision.transforms as T
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

def MaskRCNNModel():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 3
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model

def train(trainset, testset, model, num_epochs):
    device = "cpu"

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=.005, momentum=.9, weight_decay=.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.1)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, trainset, device, epoch, print_freq=1)
        lr_scheduler.step()
        # evaluate(model, testset, device=device)

def eval(testset, model, device):
    evaluate(model, testset, device=device)

def get_transform(train):
    transform = []
    transform.append(T.ToTensor())
    return T.Compose(transform)

def collate_fn(batch):
    return tuple(zip(*batch))

def non_max_suppression(res):
    bboxes = res[0]['boxes']
    scores = res[0]['scores']
    indices = torchvision.ops.nms(boxes=bboxes, scores=scores, iou_threshold=.5)

    classes = list(res[0]['labels'])

    out_box = []
    out_class = []
    for i in indices:
       out_box.append(bboxes[i])
       out_class.append(classes[i])
    print(classes)
    print(out_class)
    return out_box, out_class


def driver(property):

    transformer = get_transform(None)
    trainset = trash_can_coco_torch.CocoDataset(dataset_dir='../instance_version_padded',
                                                img_path='train/',
                                                ann_path='instances_train_trashcan.json',
                                                prop_file='../mappings/' + property + '.json',
                                                transform=transformer)
    testset = trash_can_coco_torch.CocoDataset(dataset_dir='../instance_version_padded',
                                               img_path='val/',
                                               ann_path='instances_val_trashcan.json',
                                               prop_file='../mappings/' + property + '.json',
                                               transform=transformer)

    trainset = trash_can_coco_torch.remove_images_without_annotations(trainset)

    valid = []

    try:
        fp = open('valid.txt')
        valid = fp.read()[1:-1]
        valid = valid.split(', ')
        valid = [int(i) for i in valid]
        fp.close()
    except:
        for i in range(len(trainset)):
            im, target = trainset.__getitem__(i)
            bbox = target['boxes']
            if i % 500 == 0:
                print(i, '/', len(trainset))
            if bbox.size(dim=0) > 0:
                valid += [i]
        fp = open('valid.txt', 'w+')
        fp.write(str(valid))
        fp.close()

    trainset = torch.utils.data.Subset(trainset, valid)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    model = MaskRCNNModel()
    try:
        l = torch.load('../models/' + property + '.pth', encoding='ascii')
        model.load_state_dict(l)
        print("loaded model")
    except:
        print("no model found")

    # train(trainset=trainloader, testset=testloader, model=model, num_epochs=1)
    visualize(model, testset.__getitem__(9)[0])
    torch.save(model.state_dict(), '../models/' + property + '.pth')


def visualize(model, im):
    model.to("cpu")
    model.eval()
    res = model([im])
    im = im.permute(1,2,0).detach().numpy()
    im *= 255
    im = np.ascontiguousarray(im, dtype=np.uint8)
    bboxes, classes = non_max_suppression(res)
    for count, i in enumerate(bboxes):
        pt1 = i[:2].detach().numpy().astype('uint32')
        pt2 = i[2:].detach().numpy().astype('uint32')
        if classes[count] == 1:
            im = cv2.rectangle(im, pt1, pt2, color=(0, 255,0))
            cv2.putText(im, 'metal', (pt1[0], pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36, 255, 12), 1)
        else:
            im = cv2.rectangle(im, pt1, pt2, color=(255, 0, 0))
            cv2.putText(im, 'non metal', (pt1[0], pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36, 255, 12), 1)


    plt.imshow(im)
    plt.show()

def main():
    driver('biological')
    # driver('metal')
if __name__ == '__main__':
    main()