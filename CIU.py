from typing import Any
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
import pandas as pd
import cv2
import numbers
import numpy as np
np.seterr(divide="ignore", invalid="ignore")


class CIU:
    def __init__(
        self,
        model,
        out_names,
        predict_function=None,
        perturber = None,
        neutralCU = 0.5,
        debug=False,
    ):
        """
        @param model: TF/Keras model to be used.
        @param list out_names: List of output class names to be used.
        @param predict.function: Function that takes a list of images and return a numpy.ndarray with output probabilities. 
        @param perturber: Callable that takes the image and returns a list of segmentation masks and a list of lists with
        distorted images for each segmentation mask. The default is to use SlicOcclusionPerturber.
        @param neutralCU: CU value that is considered "neutral" and that provides a limit between negative and 
        positive influence in the "Contextual influence" calculation CIx(CU - neutralCU).
        @param bool debug: Displays variables for debugging purposes. The default is False.
        """

        self.model = model
        self.out_names = out_names
        self.predict_function = predict_function if predict_function is not None else model.predict_on_batch
        self.perturber = perturber
        self.neutralCU = neutralCU
        self.debug = debug

        if self.perturber is None:
            self.perturber = SlicOcclusionPerturber()

        # Set internal object variables to default values.
        self.original_image = None
        self.image = None
        self.segments = None
        self.ciu_segments = None

    # The method to call for getting CIU values for all superpixels. It returns a 'dict' object. 
    def Explain(self, image):
        """
        @param image: The image to be explained
        """ 
        
        # Memorize image, store shape also
        self.original_image = image
        self.image = np.copy(self.original_image)
        self.image_shape = self.image.shape
        
        # Necessary reshape for model predict, not so nice to have that here, 
        # should still be abstracted away somehow.
        dnn_img = self.image.reshape(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])
        
        segment_masks, perturbed_images = self.perturber(dnn_img)
        segment_masks = segment_masks[0] #TODO: Hardcoded to work with 1 image for now
        perturbed_images = perturbed_images[0]
        nbr_segments = len(segment_masks)
        self.segments = segment_masks

        # Initialise CIU result, begin with current output.
        outvals = self.predict_function(dnn_img)
        nbr_outputs = outvals.shape[1]
        ci_s = np.zeros((nbr_outputs, nbr_segments))
        cu_s = np.zeros((nbr_outputs, nbr_segments))
        cinfl_s = np.zeros((nbr_outputs, nbr_segments))
        cmins = np.zeros((nbr_outputs, nbr_segments))
        cmaxs = np.zeros((nbr_outputs, nbr_segments))

        # Go through all superpixels and calculate CIU values.
        for i in range(0, nbr_segments):
            predictions = self.predict_function(perturbed_images[i])
            sp_vals = np.stack((outvals, predictions), axis=1)
            cmins[:,i] = sp_vals.min(axis=1)
            cmaxs[:,i] = sp_vals.max(axis=1)
            diff = cmaxs[:,i] - cmins[:,i]
            ci = diff
            cu = 0.5 if diff.any() == 0 else (outvals - cmins[:,i])/diff
            #if strategy == "inverse":
            #    ci = 1 - ci
            ci_s[:,i] = ci
            cu_s[:,i] = cu
            cinfl_s[:,i] = ci_s[:,i]*(cu_s[:,i] - self.neutralCU)

        cu_s = np.nan_to_num(cu_s, nan=0.5)

        self.ciu_segments = {
            "outnames": self.out_names,
            "outvals": outvals,
            "CI": ci_s,
            "CU": cu_s,
            "Cinfl": cinfl_s,
            "cmin": cmins,
            "cmax": cmaxs,
        }

        return self.ciu_segments
    
    # Method that returns a version of the explained image that shows only superpixels
    # with CI values equal or over "CI_limit" and CU values equal or over "CU_limit".
    def ImageInfluentialSegmentsOnly(self, ind_output=0, Cinfl_limit=None, type = "why", CI_limit=0.5, CU_limit=0.51):
        # Do based on CI and CU
        if Cinfl_limit is None:
            ci_s = self.ciu_segments["CI"][ind_output,:]
            cu_s = self.ciu_segments["CU"][ind_output,:]
            sp_ind = np.where((ci_s < CI_limit) | (cu_s < CU_limit))
        else:
            cinfl_s = self.ciu_segments["Cinfl"][ind_output,:]
            if type == "why":
                sp_ind = np.where(cinfl_s < Cinfl_limit)
            elif type == "whynot":
                sp_ind = np.where(cinfl_s > Cinfl_limit)
            else:
                raise ValueError("Unknown explanation type")
            
        res_image = self.make_superpixels_transparent(sp_ind[0])#TODO: Currently hardcoded for one image, make general in future
        return res_image

    def make_superpixels_transparent(self, sp_array, bg_color=(190,190,190)):
        bg_color = np.array(bg_color if issubclass(self.original_image.dtype.type, numbers.Integral) else bg_color/255)
        fudged_image = np.copy(self.original_image)
        for x in sp_array:
            fudged_image[self.superpixels == x] = bg_color
        return fudged_image


class SuperpixelPerturber:
    def __init__(self, segmenter, strategy, distortion):
        """
        @param segmenter: Callable that takes an image and produces a list of np.arrays of masks for each segment.
        @param strategy: Callable that takes an image and a list of segment masks and returns a list with lists with distortion masks for each segment
        @param distortion: Callable that take an image and a list of lists of distortion masks and returns the same lists with the image distorited in the masked areas.
        """
        
        self.segmenter = segmenter
        self.strategy = strategy #TODO: In the future, consider making strategy masks define the strenght (between 1 and 0) of which a distortion is overlaid the original image.
        self.distortion = distortion

    def __call__(self, image, segmenter_kwds={}, strategy_kwds={}, distortion_kwds={}):
        """
        @param image: Image to return perturbed versions of for each segment.
        @param segmenter_kwds: Dict with additional keyword arguments used by the segmenter.
        @param strategy_kwds: Dict with additional keyword arguments used by the strategy.
        @param distortion_kwds: Dict with additional keyword arguments used by the distortion.
        """

        #Clone image to not edit original
        image = np.copy(image)

        #Segment the image
        segment_masks = self.segmenter(image, **segmenter_kwds)

        #Create indices for where the images should be pertubed for each superpixel
        pertubation_masks = self.strategy(image, segment_masks, **strategy_kwds)

        #Create pertubed images for each segment
        perturbed_images = self.distortion(image, pertubation_masks)
        
        return (segment_masks, perturbed_images)


class SlicSegmenter:
    def __init__(self, nbr_segments, compactness):
        """
        @param int nbr_segments: The amount of target segments to be used by the SLIC algorithm. The default is 50.
        @param int compactness: The compactness of the segments accounting for proximity or RGB values. The default is 10
        and logarithmic.
        """
        self.nbr_segments = nbr_segments
        self.compactness = compactness

    def __call__(self, image):
        segments = slic(
            image,
            n_segments=self.nbr_segments,
            compactness=self.compactness,
            start_label=0,
        )
        segment_masks = []
        for image_segments in segments:
            image_masks = []
            segment_ids = np.unique(image_segments)
            for id in segment_ids:
                image_masks.append(np.where(image_segments==id,1,0))
            segment_masks.append(np.array(image_masks))
        return segment_masks


class EntireSegmentStrategy:
    def __init__(self, inverse=False):
        """
        @param bool inverse: Whether to mask every pixel except in the current segment or every pixel in the current segment
        """
        self.inverse = inverse

    def __call__(self, image, segment_masks):
        distortion_masks = []
        if self.inverse:
            for image_masks in segment_masks:
                distortion_masks.append(np.expand_dims(np.where(image_masks==0,1,0),axis=1))
        else: 
            for image_masks in segment_masks:
                distortion_masks.append(np.expand_dims(image_masks,axis=1))
        return distortion_masks


class SingleColorDistortion:
    def __init__(self, background_color):
        """
        @param background_color: Background color to use for "transparent", in RGB. The default is gray (190, 190, 190).
        """
        self.background_color = np.array(background_color)
        self.background_color_float = self.background_color/255

    def __call__(self, image, distortion_masks):
        #Detect whether the image is in [0,1] or a  in [0,1,..,255]
        color = self.background_color if issubclass(image.dtype.type, numbers.Integral) else self.background_color_float

        distorted_images = []
        for id, image_distortion_masks in enumerate(distortion_masks):
            indices = np.nonzero(image_distortion_masks)
            distorted_image = np.tile(np.copy(image[id]), (image_distortion_masks.shape[0],image_distortion_masks.shape[1],1,1,1))
            distorted_image[indices] = color
            distorted_images.append(distorted_image)
        return distorted_images


def SlicOcclusionPerturber(background_color=(190,190,190), strategy="straight", nbr_segments=50, compactness=10):
    """
    @param background_color: Background color to use for "transparent", in RGB. The default is gray (190, 190, 190). 
    @param bool strategy: Defines whether the CIU strategy should be "inverse" or "straight". The default is "straight".
    @param int nbr_segments: The amount of target segments to be used by the SLIC algorithm. The default is 50.
    @param int compactness: The compactness of the segments accounting for proximity or RGB values. The default is 10 and logarithmic.
    """
    segmenter = SlicSegmenter(nbr_segments, compactness)
    strategy = EntireSegmentStrategy(inverse=strategy=="inverse")
    distortion = SingleColorDistortion(background_color)
    return SuperpixelPerturber(segmenter, strategy, distortion)