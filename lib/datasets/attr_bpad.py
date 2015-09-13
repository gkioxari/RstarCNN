# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) Microsoft. All rights reserved.
# Written by Ross Girshick, 2015.
# Licensed under the BSD 2-clause "Simplified" license.
# See LICENSE in the project root for license information.
# --------------------------------------------------------
# --------------------------------------------------------
# R*CNN
# Written by Georgia Gkioxari, 2015.
# See LICENSE in the project root for license information.
# --------------------------------------------------------

import datasets.pascal_voc
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import pdb

class attr_bpad(datasets.imdb):
    def __init__(self, image_set, devkit_path=None):
        datasets.imdb.__init__(self, 'bpad_' + image_set)
        self._year = '2015'
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._base_path = os.path.join(self._devkit_path, 'BAPD')
        self._classes = ('is_male', 'has_long_hair', 'has_glasses',
                         'has_hat', 'has_tshirt', 'has_long_sleeves',  
                         'has_shorts', 'has_jeans', 'has_long_pants')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._base_path), \
                'Path does not exist: {}'.format(self._base_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._base_path, 'Images',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._base_path, 'selective_search',
                                      'ss_attributes_' + self._image_set + '.mat')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        raw_data = sio.loadmat(image_set_file)
        images = raw_data['images'].ravel()
        image_index = [im[0].strip() for im in images]

        return image_index

    def _get_default_path(self):
        """
        Return the default path where data is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data')


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        # Load all annotation file data (should take < 30 s).
        gt_roidb = self._load_annotation()

        # print number of ground truth classes
        cc = np.zeros(len(self._classes), dtype = np.int16)
        for i in xrange(len(gt_roidb)):
            gt_classes = gt_roidb[i]['gt_classes']
            num_objs = gt_classes.shape[0]
            for n in xrange(num_objs):
                valid_classes = np.where(gt_classes[n] == 1)[0]
                cc[valid_classes] +=1

        for ic,nc in enumerate(cc):
            print "Count {:s} : {:d}".format(self._classes[ic], nc)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                 self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_roidb(gt_roidb)
        roidb = self._merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _merge_roidbs(self, a, b):
        assert len(a) == len(b)
        for i in xrange(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.vstack((a[i]['gt_classes'],
                                            b[i]['gt_classes']))
            a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                       b[i]['gt_overlaps']])
        return a

    def _load_selective_search_roidb(self, gt_roidb):

        filename = os.path.join(self._base_path, 'selective_search',
                                      'ss_attributes_' + self._image_set + '.mat')

        # filename = op.path.join(self.cache_path, 'MCG_data', self.name + '.mat')

        assert os.path.exists(filename), \
                'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)

        num_images = raw_data['boxes'].ravel().shape[0]

        ss_roidb = []
        for i in xrange(num_images):
            boxes = raw_data['boxes'].ravel()[i][:, (1, 0, 3, 2)] - 1
            num_boxes = boxes.shape[0]
            gt_boxes = gt_roidb[i]['boxes']
            num_objs = gt_boxes.shape[0]
            gt_classes = gt_roidb[i]['gt_classes']
            gt_overlaps = \
                    utils.cython_bbox.bbox_overlaps(boxes.astype(np.float),
                                                    gt_boxes.astype(np.float))

            overlaps = scipy.sparse.csr_matrix(gt_overlaps)

            ss_roidb.append({'boxes' : boxes,
                             'gt_classes' : np.zeros((num_boxes, self.num_classes),
                                                      dtype=np.int32),
                             'gt_overlaps' : overlaps,
                             'flipped' : False})
        return ss_roidb

    def _load_annotation(self):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """

        gt_roidb = []

        filename = os.path.join(self._base_path, 'ground_truth',
                                      'gt_attributes_' + self._image_set + '.mat')

        assert os.path.exists(filename), \
                'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename, mat_dtype=True)

        all_boxes = raw_data['boxes'].ravel()
        all_images = raw_data['images'].ravel()
        all_attributes = raw_data['attributes'].ravel()

        num_images = len(all_images)
        for imi in xrange(num_images):
            num_objs = all_boxes[imi].shape[0]
            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_objs, self.num_classes), dtype=np.int32)
            overlaps = np.zeros((num_objs, num_objs), dtype=np.float32)
            
            # Load object bounding boxes into a data frame.
            for i in xrange(num_objs):
                # Make pixel indexes 0-based
                box = all_boxes[imi][i]
                assert(not np.any(np.isnan(box)))
                
                # Read attributes labels
                attr = all_attributes[imi][i]
                
                # Change attributes labels
                # -1 -> 0
                # 0 -> -1
                unknown_attr = attr == 0
                neg_attr = attr == -1
                attr[neg_attr] = 0
                attr[unknown_attr] = -1

                boxes[i, :] = box - 1
                gt_classes[i, :] = attr
                overlaps[i, i] = 1.0

            overlaps = scipy.sparse.csr_matrix(overlaps)
            
            gt_roidb.append({'boxes' : boxes,
                             'gt_classes': gt_classes,
                             'gt_overlaps' : overlaps,
                             'flipped' : False})

        return gt_roidb

    def _write_results_file(self, all_boxes, comp):
        
        path = os.path.join(self._devkit_path, 'results', 'BAPD')
            
        print 'Writing results file'.format(cls)
        filename = path + comp + '.txt'
        with open(filename, 'wt') as f:
            for i in xrange(all_boxes.shape[0]):
                ind = all_boxes[i,0].astype(np.int64)
                index = self.image_index[ind-1]
                voc_id = all_boxes[i,1].astype(np.int64)

                f.write('{:s} {:d}'.format(index, voc_id))
                for cli in xrange(self.num_classes):
                    score = all_boxes[i,2+cli]
                
                    f.write(' {:.3f}'.format(score))   
                f.write('\n')        

if __name__ == '__main__':
    d = datasets.pascal_voc('trainval', '2012')
    res = d.roidb
    from IPython import embed; embed()
