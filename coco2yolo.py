import logging
from pylabel import importer
import os

path_test = '../instance_version_padded/instances_val_trashcan.json'

path_test_im = '../instance_version_padded/test'

dataset = importer.ImportCoco(path_test, path_to_images=path_test_im, name="trashCan_YOLO")
dataset.path_to_annotations = "instance_version_padded/yolo"
dataset.export.ExportToYoloV5()