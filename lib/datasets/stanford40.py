# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) Microsoft. All rights reserved.
# Written by Ross Girshick, 2015.
# Licensed under the BSD 2-clause "Simplified" license.
# See LICENSE in the project root for license information.
# --------------------------------------------------------

import datasets.stanford40
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

class stanford40(datasets.imdb):
    def __init__(self, image_set, devkit_path=None):
        datasets.imdb.__init__(self, 'stanford40_' + image_set)
        self._year = '2015'
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._base_path = os.path.join(self._devkit_path, 'Stanford40')
        self._classes = self.get_classes()
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                            'use_salt' : True}

        assert os.path.exists(self._devkit_path), \
                'devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._base_path), \
                'Base Path does not exist: {}'.format(self._base_path)

    def get_classes(self):
        """ 
        Read the action classes and return them
        """
        action_file = os.path.join(self._base_path, 'actions.txt')
        assert os.path.exists(action_file), \
           'Actions log does not exist: {}'.format(action_path)

        actions = []
        with open(action_file, 'rb') as fid:
            #dummy 1st line
            l = fid.readline()
            for line in fid:
                c = line.split('\t')
                actions.append(c[0])

        assert len(actions)==40
        return actions

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._base_path, 'JPEGImages',
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
                                      'selective_search_' + self._image_set + '.mat')
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
                valid_classes = gt_classes[n] 
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
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                            b[i]['gt_classes']))
            a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                       b[i]['gt_overlaps']])
        return a

    def _load_selective_search_roidb(self, gt_roidb):

        filename = os.path.join(self._base_path, 'selective_search',
                                      'selective_search_' + self._image_set + '.mat')

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

            argmaxes = gt_overlaps.argmax(axis=1)
            maxes = gt_overlaps.max(axis=1)
            I = np.where(maxes > 0)[0]
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)
            overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            overlaps = scipy.sparse.csr_matrix(overlaps)
            ss_roidb.append({'boxes' : boxes,
                                      'gt_classes' : -np.ones((num_boxes,),
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
                                      'gt_' + self._image_set + '.mat')

        assert os.path.exists(filename), \
                'Gt data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename, mat_dtype=True)

        all_boxes = raw_data['boxes'].ravel()
        all_images = raw_data['images'].ravel()
        all_images = [im[0].strip() for im in all_images]

        num_images = len(all_images)
        for imi in xrange(num_images):
            num_objs = all_boxes[imi].shape[0]
            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_objs,), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

            # get class by image name
            image_name = all_images[imi]
            slash_ind = image_name.rfind('_')
            action_name = image_name[:slash_ind]
            action_ind = self._class_to_ind[action_name]
            
            # Load object bounding boxes into a data frame.
            for i in xrange(num_objs):
                # Make pixel indexes 0-based
                box = all_boxes[imi][i]
                assert(not np.any(np.isnan(box)))
                
                boxes[i, :] = box - 1
                gt_classes[i] = action_ind
                overlaps[i, action_ind] = 1.0

            overlaps = scipy.sparse.csr_matrix(overlaps)
            
            gt_roidb.append({'boxes' : boxes,
                                      'gt_classes': gt_classes,
                                      'gt_overlaps' : overlaps,
                                      'flipped' : False})

        return gt_roidb

    def _ap(self, all_scores, all_labels):
        ap = np.zeros((self.num_classes), dtype = np.float32)
        
        for a in xrange(self.num_classes):
            tp = all_labels[self.classes[a]]==a
            npos = np.sum(tp, axis = 0);
            fp = all_labels[self.classes[a]]!=a
            sc = all_scores[self.classes[a]]
            cat_all = np.hstack((tp,fp,sc))
            ind = np.argsort(cat_all[:,2])
            cat_all = cat_all[ind[::-1],:]
            tp = np.cumsum(cat_all[:,0], axis = 0);
            fp = np.cumsum(cat_all[:,1], axis = 0);

            # # Compute precision/recall
            rec = tp / npos;
            prec = np.divide(tp, (fp+tp));
            ap[a] = self.VOCap(rec, prec);   

        print 'Mean AP = %.2f'%(np.mean(ap)*100)  
        return ap

    def VOCap(self, rec, prec):
        rec = rec.reshape(rec.size,1); prec = prec.reshape(prec.size,1)
        z = np.zeros((1,1)); o = np.ones((1,1));
        mrec = np.vstack((z, rec, o))
        mpre = np.vstack((z, prec, z))
        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])

        # i = find(mrec(2:end)~=mrec(1:end-1))+1;
        I = np.where(mrec[1:] != mrec[0:-1])[0]+1;
        ap = 0;
        for i in I:
            ap = ap + (mrec[i] - mrec[i-1])*mpre[i];
        return ap 

if __name__ == '__main__':
    d = datasets.pascal_voc('trainval', '2012')
    res = d.roidb
    from IPython import embed; embed()
