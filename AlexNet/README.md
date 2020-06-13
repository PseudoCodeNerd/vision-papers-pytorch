## PyTorch implementation of AlexNet

Goal of this implementation is to gain deeper insight into the 2012 breakthrough Computer Vision paper, *'ImageNet Classification with Deep Convolutional Neural Networks'* by Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton and to get acquainted with the structure of deep learning projects.

---

### Table of Contents
1. [Code](#code)
   - [Requirements](#requirements)
   - [Usage](#usage)
2. [Background](#background)
3. [Architecture & Features](#architecturefeatures)
4. [Final Thoughts](#final-thoughts)

#### Code

##### Requirements

Clone environment used to build this by `pip install -r requirements.txt`.

Other, high-level requirements are :

- **Python >= 3.5.0**
- **PyTorch >= 0.4.0** and its accompanying packages.
  - CUDA is highly recommended.
- This implemenation uses the ILSVRC 2012 dataset, also known as the 'ImageNet 2012 dataset'. The data size is dreadfully large (138G!)
  - Download ImageNet data (in the form of `ILSVRC2012_img_train`) from [here](http://www.image-net.org/challenges/LSVRC/2012/)
  -  Extract data using [`extract_data.sh`](extract_data.sh), check [`setup_data.py`](setup_data.py) to see if train and val data dirs are correct.
  -  Download pretrained model weights from [https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth](https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth) and place it in `~/.torch/models` .
  -  Move validation images to labeled subfolders using `setup_data.sh` to *setup* the data.
  - `setup_data.sh` courtesy of [Soumith Chintala](https://github.com/soumith/imagenetloader.torch).

#### Usage

**Training**

`python main.py [folder with train and val datasets] -a alexnet --lr 0.01`

**More Possibilities**

```
usage: main.py [-h] [-a ARCH] [--epochs N] [--start-epoch N] [-b N] [--lr LR]
               [--momentum M] [--weightdecay W] [-j N] [-p] [-e]
               DIR

PyTorch AlexNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: alexnet (default: alexnet)
  --epochs N            total epochs to run
  -b N, --batch-size N  mini-batch size / default:256
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum for optimising process
  --weightdecay W, --wd W
                        Weight decay / default: 1e-4'
  -j N, --workers N     no. of available gpu workers / default:1
  -p, --pretrained      use pre-trained weights
  -e, --evaluate        Evaluate model performance on validation set


```

---

#### Background

It is believed among CV academia that this paper (AlexNet) is the one which brought about the CNN revolution in the field of Computer Vision (some may argue that it's [Yann LeCun's 1998 *LeNet*](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) which deserves this accolade).
Cited 21,409 times, this paper is widely regarded as one of the most influential publications. The architecture/approach put forward by the authors was able to win the 2012 ImageNet Large-Scale Visual Recognition Challenge *a.k.a* the ultimate computer vision challenge  where teams of researchers compete to classify, detect and localize images  with the most accuracy (or least error). 
Winning with Top-5 error rate of 15.4%, with the next best accurate submission having a T5-ER of 26.2%, Convolutional Neural Networks instantly became the *go-to* approach for CV tasks after this paper.

#### Architecture/Features

![Model Architecture](https://i.ibb.co/0s7Wwrn/image.png)
<center><i>Image from the original paper</i></center><br>

> An architecture visualisation I really like is [this one](https://neurohive.io/wp-content/uploads/2018/10/AlexNet-1.png.)

*Note: Due to the non-availabilty of powerful GPUs back in the day, the authors split the training process onto 2 different GPUs. Hence, the 2 streams in the above image.*

Given an image of dims. $227*227*3$, the model's task is to classify it into $1000$ possible categories of images (read more about ImageNet [here](http://www.image-net.org/about-overview)). 

The architecture seems simple by today's standards (thank InceptionNet and ResNets) but it was a very complex (and deep) architecture by 2012 standards. ReLU was used as the activation function for each layer, since it was observed that it was [remarkably faster](https://datascience.stackexchange.com/a/23502/80844) than tradition *tanH* function.

We have 5 convolutional layer, each followed by a max pooling and dropout layers. After each complete layer  (conv-->pooling-->dropout), the image dimensions are reduced while the number of filters increased. Later, the volume is unrolled into one vector and after application of 2 Fully Connected Layers, we have a vector of size 1000 to classify the image into the 1000 classes of the ImageNet datase (gives probabilities of each class using the [softmax operation](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)).

In total, this particular architecure has ~$60$ million parameters to be trained. Due to the scale of the data and this large no. of params, it was natural for the final trained model to be overfitting on the test data. To combat this, the authors used [*Dropout*](http://www.jmlr.org/papers/volume15/srivastava14a.old/source/srivastava14a.pdf) and *Data Augmentation* techniques (consisting of image translations, horizontal reflections, and patch extractions.)

#### Final Thoughts

AlexNet's win was the coming out party for CNNs in the computer vision community. This was the first time a model performed so well on a historically difficult ImageNet dataset. Utilizing techniques like data augmentation and dropout this paper really illustrated the benefits of CNNs and backed them up with record breaking performance in the competition.