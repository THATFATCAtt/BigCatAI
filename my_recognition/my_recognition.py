#!/usr/bin/python3
import jetson_inference
import jetson_utils


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
opt = parser.parse_args()
img = jetson_utils.loadImage(opt.filename)
net = jetson_inference.imageNet(opt.network)
class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)



if "leopard" in class_desc or "Leopard" in class_desc:
    print("This is a Leopard. They are uniqe in the fact that they like chilling in trees, and are also the smallest of the big cats. The spots on a leopard are actually called rosettes. If you encounter a Leopard in the wild, DO NOT RUN as it will instinctivley chase you. Instead, you should make loud noises by clapping your hands, shouting and wave your arms and fight back if neccassary.")


if "cheetah" in class_desc or "Cheetah" in class_desc:
    print("This is a Cheetah. They are the fastest land animals on Earth, have about 2000 spots on their bodies, and are unable to roar due to their skelatal structure. If you encounter a Cheetah in the wild. DO NOT RUN as it will instictiveley chase you. Instead, you should try to intimidate the cheetah or back away slowly while making eye contact.")


print("image is recognized as "+ str(class_desc) +" (class #"+ str(class_idx) +") with " + str(confidence*100)+"% confidence")


