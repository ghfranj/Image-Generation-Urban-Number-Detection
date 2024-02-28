# Image-Generation-Urban-Number-Detection

1- Training (or retraining) a neural network to detect numbers in images (a number is a sequence of up to 20 characters in length from numbers, upper and lowercase letters of the Russian or English alphabet, hyphens and slashes).
2- Taking a photo of the license plates on the city streets (not from the Internet, let’s check by searching the image) and run your algorithm on them.
3- The detection quality is assessed using the IoU, Precision, Recall, mAP metrics on the test part of The Street View House Numbers set.

Steps:
1- Defining model: Faster RCNN
~~~
def get_object_detection_model(num_classes):
    # load a model pre-trained pre-trained on COCO

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
~~~
2- Dataset used: SVHN housenumbers dataset
~~~
train_url = "http://ufldl.stanford.edu/housenumbers/train.tar.gz"
test_url = "http://ufldl.stanford.edu/housenumbers/test.tar.gz"
~~~
3- Testing a photo of the license plates on the city streets:
![image](https://github.com/ghfranj/Image-Generation-Urban-Number-Detection/assets/98123238/7e681f9a-dc22-4328-82eb-853d327a7a2b)

