# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# --------------------------------------------------------
# R*CNN
# Written by Georgia Gkioxari, 2015.
# See LICENSE in the project root for license information.
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""

import numpy as np
from fast_rcnn.config import cfg
import utils.cython_bbox

def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        gt_classes = roidb[i]['gt_classes']

        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        class_inds = gt_overlaps.argmax(axis=1)
        I = np.where(max_overlaps>0.0)[0]
        # gt class that had the max overlap
        max_classes = np.zeros((gt_overlaps.shape[0], imdb.num_classes), dtype = np.float32)
        for j in xrange(len(I)):
            max_classes[I[j]] = gt_classes[class_inds[I[j]]]

        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps