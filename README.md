# R\*CNN
Source code for R\*CNN, created by Georgia Gkioxari at UC Berkeley.

[![Gitter](https://badges.gitter.im/gkioxari/RstarCNN.svg)](https://gitter.im/gkioxari/RstarCNN?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

### Introduction

R\*CNN was initialiy described in an [arXiv tech report] (http://arxiv.org/abs/1505.01197)

### License 

R\*CNN is released under the BSD License

### Citing R\*CNN

If you use R\*CNN, please consider citing:

	@article{rstarcnn2015,
        Author = {G. Gkioxari and R. Girshick and J. Malik},
        Title = {Contextual Action Recognition with R\*CNN},
        Booktitle = {ICCV},
        Year = {2015}
    }

### Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Downloads](#downloads)

### Requirements

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe 	installation instructions](http://caffe.berkeleyvision.org/installation.html))

	**Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1

  ```
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`

### Installation

1. Clone the RstarCNN repository
  	```Shell
  	# Make sure to clone with --recursive
  	git clone --recursive https://github.com/gkioxari/RstarCNN.git
  	```

2. Build the Cython modules
 	``` Shell
 	cd $ROOT/lib
 	make
 	```

3. Build Caffe and pycaffe
	```Shell
    cd $ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

### Usage

Train a R\*CNN classifier. For example, train a VGG16 network on VOC 2012 trainval:

```Shell
./tools/train_net.py --gpu 0 --solver models/VGG16_RstarCNN/solver.prototxt \
	--weights reference_models/VGG16.v2.caffemodel
```

Test a R\*CNN classifier

```Shell
./tools/test_net.py --gpu 0 --def models/VGG16_RstarCNN/test.prototxt \
	--net output/default/voc_2012_trainval/vgg16_fast_rstarcnn_joint_iter_40000.caffemodel
```

### Downloads

1. PASCAL VOC 2012 Action Dataset

	Place the VOCdevkit2012 inside the `$ROOT/data` directory

	Download the selective search regions for the images from [here](https://www.cs.berkeley.edu/~gkioxari/RstarCNN/ss_voc2012.tar.gz) and place them inside the `$ROOT/data/cache` directory

2. Berkeley Attributes of People Dataset

	Download the data from [here](https://www.cs.berkeley.edu/~gkioxari/RstarCNN/BAPD.tar.gz) and place them inside the `$ROOT/data` directory

3. Stanford 40 Dataset
      
      Download the data from [here](https://www.cs.berkeley.edu/~gkioxari/RstarCNN/Stanford40.tar.gz) and place them inside `$ROOT/data` directory. R*CNN achieves 90.85% on the test set (trained models provided in 5)

4. Reference models
	
	Download the VGG16 reference model trained on ImageNet from [here](https://www.cs.berkeley.edu/~gkioxari/RstarCNN/reference_models.tar.gz) (500M)

5. Trained models
	
	Download the models as described in the paper from [here](https://www.cs.berkeley.edu/~gkioxari/RstarCNN/trained_models.tar.gz) (3.6G)



  