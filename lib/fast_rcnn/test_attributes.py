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

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from utils.cython_nms import nms
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os
import scipy.io as sio
import utils.cython_bbox

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, gt_rois, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None, 'secondary_rois': None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    blobs['rois'] = _get_rois_blob(gt_rois, im_scale_factors)
    blobs['secondary_rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(net, im, gt_boxes, ss_boxes):
    """Detect classes in an image given object proposals.
    """ 
    blobs, unused_im_scale_factors = _get_blobs(im, gt_boxes, ss_boxes)
    base_shape = blobs['data'].shape

    blobs_rois = blobs['rois'].astype(np.float32, copy=False)
    blobs_rois = blobs_rois[:, :, np.newaxis, np.newaxis]

    blobs_sec_rois = blobs['secondary_rois'].astype(np.float32, copy=False)
    blobs_sec_rois = blobs_sec_rois[:, :, np.newaxis, np.newaxis]

    # reshape network inputs
    net.blobs['data'].reshape(base_shape[0], base_shape[1],
                              base_shape[2], base_shape[3])
    net.blobs['rois'].reshape(blobs_rois.shape[0], 5, 1, 1)
    net.blobs['secondary_rois'].reshape(blobs_sec_rois.shape[0], 5, 1, 1)
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                            rois=blobs_rois,
                            secondary_rois = blobs_sec_rois)

    scores = blobs_out['cls_score']
    secondary_scores = blobs_out['context_cls_score']

    # Compute overlap
    boxes_overlaps = \
      utils.cython_bbox.bbox_overlaps(ss_boxes.astype(np.float), 
                                      gt_boxes.astype(np.float))

    selected_boxes = np.zeros((scores.shape[1], 4, gt_boxes.shape[0]))

    # Sum of Max
    for i in xrange(gt_boxes.shape[0]):
        keep_inds = np.where((boxes_overlaps[:,i]>=cfg.TEST.IOU_LB) &
                             (boxes_overlaps[:,i]<=cfg.TEST.IOU_UB))[0]
        if keep_inds.size > 0:
            this_scores = np.amax(secondary_scores[keep_inds,:], axis=0)
            scores[i,:] = scores[i,:]+this_scores
            winner_ind  = np.argmax(secondary_scores[keep_inds,:], axis=0)            
            selected_boxes[:,:,i] = ss_boxes[keep_inds[winner_ind]]
       
    # Sigmoid    
    scores = np.exp(-scores)
    scores = 1.0 / (1.0 + scores)

    return scores, secondary_scores, selected_boxes

def vis_detections(im, boxes, scores, classes):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(1):
        bbox = boxes[i, :4]
        sscore = scores[i, :]

        for cli in xrange(len(classes)):
            print "{:s} = {:.3f}".format(classes[cli], sscore[cli])
        
        #plt.cla()
        plt.imshow(im)
        plt.gca().add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                           bbox[2] - bbox[0],
                           bbox[3] - bbox[1], fill=False,
                          edgecolor='r', linewidth=3)
                )
        plt.show()

def test_net(net, imdb):
    """Test a R*CNN network on an image database."""
    num_images = len(imdb.image_index)
    num_classes = imdb.num_classes

    all_boxes = np.zeros((0, 2+num_classes), dtype = np.float32)
    all_selected_boxes = np.zeros((num_classes, 4, 0))

    # timers
    _t = {'im_detect' : Timer()}

    roidb = imdb.roidb
    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))

        num_gt = roidb[i]['gt_overlaps'].shape[1]
        gt_boxes = roidb[i]['boxes'][0:num_gt]
        ss_boxes = roidb[i]['boxes'][num_gt:]

        _t['im_detect'].tic()
        scores, secondary_scores, selected_boxes = im_detect(net, im, gt_boxes, ss_boxes)
        _t['im_detect'].toc()

        # Visualize detections
        # vis_detections(im, gt_boxes, scores, imdb.classes)

        for j in xrange(gt_boxes.shape[0]):
            # store image id and voc_id (1-indexed)
            temp = np.array([i+1, j+1], ndmin=2)
            temp = np.concatenate((temp, np.array(scores[j,:],ndmin=2)), axis=1)
            all_boxes = np.concatenate((all_boxes, temp), axis=0)

        all_selected_boxes = np.concatenate((all_selected_boxes, selected_boxes), axis = 2)

        print 'im_detect: {:d}/{:d} {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time)

    

    print 'Writing VOC results'
    imdb._write_results_file(all_boxes, 'attributes_joint')
