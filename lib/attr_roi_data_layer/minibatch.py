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

"""Compute minibatch blobs for training a R*CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
import utils.cython_bbox
# import pdb

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    # Now, build the region of interest and label blobs
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    secondary_rois_blob = np.zeros((0, 5), dtype = np.float32)
    labels_blob = np.zeros((0, num_classes), dtype=np.float32)
    
    for im_i in xrange(num_images):
        labels, overlaps, im_rois, sec_rois \
            = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                           num_classes)
        # Add to RoIs blob
        rois = _project_im_rois(im_rois, im_scales[im_i])
        batch_ind = im_i * np.ones((rois.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, rois))
        rois_blob = np.vstack((rois_blob, rois_blob_this_image))

        # Add to Secondary RoIs blob
        secondary_rois = _project_im_rois(sec_rois, im_scales[im_i])
        secondary_batch_ind = im_i * np.ones((secondary_rois.shape[0], 1))
        secondary_blob_this_image = np.hstack((secondary_batch_ind, secondary_rois))
        secondary_rois_blob = np.vstack((secondary_rois_blob, secondary_blob_this_image))

        # Add to labels, bbox targets, and bbox loss blobs
        labels_blob = np.vstack((labels_blob, labels))

    # Size checks
    assert(secondary_rois_blob.shape[0]==cfg.TRAIN.CONTEXT_NUM_ROIS*rois_blob.shape[0]),"Context and ROIs don't match"
    assert(labels_blob.shape[0]==rois_blob.shape[0]),"Labels and ROIs don't match"

    # visualizations
    # _vis_minibatch(im_blob, rois_blob, secondary_rois_blob, labels_blob)

    blobs = {'data': im_blob,
             'rois': rois_blob,
             'secondary_rois': secondary_rois_blob,
             'labels': labels_blob}

    return blobs

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and context
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']
    secondary_rois = roidb['boxes']

    # overlaps of boxes
    boxes_overlaps = \
            utils.cython_bbox.bbox_overlaps(rois.astype(np.float), rois.astype(np.float))

    # Select foreground ROIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground ROIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image,
                                replace=False)

    # Select secondary ROIs for fg regions
    secondary_fg = np.zeros((0), dtype=np.int64)
    for fg_i in fg_inds:
        cinds = np.where((boxes_overlaps[:,fg_i] >= cfg.TRAIN.IOU_LB) &
                         (boxes_overlaps[:,fg_i] <= cfg.TRAIN.IOU_UB))[0]
        if cinds.size > cfg.TRAIN.CONTEXT_NUM_ROIS:
            cinds = npr.choice(cinds, size = cfg.TRAIN.CONTEXT_NUM_ROIS, 
                                    replace=False)
        elif cinds.size > 0:
            cinds = npr.choice(cinds, size = cfg.TRAIN.CONTEXT_NUM_ROIS,
                                    replace=True)

        if cinds.size > 0:
            assert(cinds.size == cfg.TRAIN.CONTEXT_NUM_ROIS),"Secondary RoIs are not of correct size"
            secondary_fg = np.concatenate((secondary_fg, cinds),axis=0)
        else:
            secondary_fg = np.concatenate((secondary_fg, np.tile(fg_i, cfg.TRAIN.CONTEXT_NUM_ROIS)), axis=0)


    assert(fg_inds.size*cfg.TRAIN.CONTEXT_NUM_ROIS == secondary_fg.size),"[FG all] Does not match"

    # The indices that we're selecting (both fg and bg)
    keep_inds = fg_inds
    keep_secondary_inds = secondary_fg
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]
    secondary_rois = secondary_rois[keep_secondary_inds]

    return labels, overlaps, rois, secondary_rois

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _vis_minibatch(im_blob, rois_blob, secondary_rois_blob, labels_blob):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        sec_rois = secondary_rois_blob[i*cfg.TRAIN.CONTEXT_NUM_ROIS:(i+1)*cfg.TRAIN.CONTEXT_NUM_ROIS,:]
        im_ind = rois[0]
        assert all(sec_rois[:,0]==im_ind)
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        attributes = labels_blob[i]

        plt.imshow(im)
        print 'attributes: ', attributes
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        for sec_i in xrange(sec_rois.shape[0]):
            plt.gca().add_patch(
            plt.Rectangle((sec_rois[sec_i,1], sec_rois[sec_i,2]), 
                           sec_rois[sec_i,3] - sec_rois[sec_i,1],
                           sec_rois[sec_i,4] - sec_rois[sec_i,2], fill=False,
                          edgecolor='g', linewidth=3)
            )   
        plt.show()
