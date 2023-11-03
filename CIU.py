from typing import Any
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
import pandas as pd
import cv2
import numpy as np
np.seterr(divide="ignore", invalid="ignore")


class CIU:
    def __init__(
        self,
        model,
        out_names,
        predict_function=None,
        background_color=(190,190,190),
        strategy = "straight",
        neutralCU = 0.5, 
        segments = None, 
        nbr_segments=50,
        compactness=10,
        debug=False,
    ):
        """
        @param model: TF/Keras model to be used.
        @param list out_names: List of output class names to be used.
        @param predict.function: Function that takes a list of images and return a numpy.ndarray with output probabilities. 
        @param background_color: Background color to use for "transparent", in RGB. The default is gray (190, 190, 190). 
        In the future this will be modified for supporting more than one different colors, patterns or other perturbation
        methods. 
        @param str strategy: Defines CIU strategy. Either "straight" or "inverse". The default is "straight".
        @param neutralCU: CU value that is considered "neutral" and that provides a limit between negative and 
        positive influence in the "Contextual influence" calculation CIx(CU - neutralCU).
        @param segments: np.array of same dimensions as image, with segment index for every pixel. The default 
        is "None", which signifies that the default SLIC method will be used for creating superpixels. 
        @param int nbr_segments: The amount of target segments to be used by the SLIC algorithm. The default is 50.
        @param int compactness: The compactness of the segments accounting for proximity or RGB values. The default is 10
        and logarithmic.
        @param bool debug: Displays variables for debugging purposes. The default is False.
        """

        self.model = model
        self.out_names = out_names
        self.predict_function = predict_function if predict_function is not None else model.predict_on_batch
        self.background_color = background_color # Easier to deal with np.array
        self.segments = segments
        self.nbr_segments = nbr_segments
        self.compactness = compactness
        self.strategy = strategy
        self.neutralCU = neutralCU
        self.debug = debug

        # Set internal object variables to default values.
        self.original_image = None
        self.image = None
        self.superpixels = None
        self.ciu_superpixels = None

    # The method to call for getting CIU values for all superpixels. It returns a 'dict' object. 
    def Explain(self, image, strategy=None):
        """
        @param str strategy: Defines CIU strategy. Either "straight" or "inverse". The default is "None", 
        which causes self.strategy to be used instead.
        """
        # Check for overriding of strategy
        if strategy is None:
            strategy = self.strategy
        
        # Memorize image, store shape also
        self.original_image = image
        self.image = np.copy(self.original_image)
        self.image_shape = self.image.shape
        
        # See if pixels are in [0,1] or in RGB and act accordingly
        bg_colour = np.array(self.background_color)/255 if self.image.dtype == 'float32' else self.background_color

        # Find superpixels, unless the segments have been provided already
        if self.segments is None:
            self.superpixels = self.segmented(self.image)
        else:
            self.superpixels = self.segments
        all_sp = np.unique(self.superpixels)
        n_sp = len(all_sp)
        
        # Necessary reshape for model predict, not so nice to have that here, 
        # should still be abstracted away somehow.
        dnn_img = self.image.reshape(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])
        
        # Initialise CIU result, begin with current output.
        outvals = self.predict_function(dnn_img)
        nbr_outputs = outvals.shape[1]
        ci_s = np.zeros((nbr_outputs, n_sp))
        cu_s = np.zeros((nbr_outputs, n_sp))
        cinfl_s = np.zeros((nbr_outputs, n_sp))
        cmins = np.zeros((nbr_outputs, n_sp))
        cmaxs = np.zeros((nbr_outputs, n_sp))

        # Create perturbed images for all superpixels. 
        # This will need to be modified when we support more perturbation options 
        # than one background color (TO BE IMPLEMENTED)!
        fudged_images = []
        for i in range(0, n_sp):
            if strategy == "inverse":
                inps = np.delete(all_sp, i)
            elif strategy == "straight":
                inps = np.array([i])
            else:
                raise ValueError("Unknown Strategy")
            fudged_image = self.perturbed_images(inps, bg_colour)
            fudged_image_reshape = fudged_image.reshape(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])
            fudged_images.append(fudged_image_reshape)

        predictions = self.predict_function(np.vstackdistortion_kwds(fudged_images))

        # Go through all superpixels and calculate CIU values.
        for i in range(0, n_sp):
            sp_vals = np.stack((outvals[0,:], predictions[i,:]), axis=1)
            cmins[:,i] = sp_vals.min(axis=1)
            cmaxs[:,i] = sp_vals.max(axis=1)
            diff = cmaxs[:,i] - cmins[:,i]
            ci = diff
            cu = 0.5 if diff.any() == 0 else (outvals - cmins[:,i])/diff
            if strategy == "inverse":
                ci = 1 - ci
            ci_s[:,i] = ci
            cu_s[:,i] = cu
            cinfl_s[:,i] = ci_s[:,i]*(cu_s[:,i] - self.neutralCU)

        cu_s = np.nan_to_num(cu_s, nan=0.5)

        self.ciu_superpixels = {
            "outnames": self.out_names,
            "outvals": outvals,
            "CI": ci_s,
            "CU": cu_s,
            "Cinfl": cinfl_s,
            "cmin": cmins,
            "cmax": cmaxs,
        }

        return self.ciu_superpixels
    
    # Method that returns a version of the explained image that shows only superpixels
    # with CI values equal or over "CI_limit" and CU values equal or over "CU_limit".
    def ImageInfluentialSegmentsOnly(self, ind_output=0, Cinfl_limit=None, type = "why", CI_limit=0.5, CU_limit=0.51):
        # Do based on CI and CU
        if Cinfl_limit is None:
            ci_s = self.ciu_superpixels["CI"][ind_output,:]
            cu_s = self.ciu_superpixels["CU"][ind_output,:]
            sp_ind = np.where((ci_s < CI_limit) | (cu_s < CU_limit))
        else:
            cinfl_s = self.ciu_superpixels["Cinfl"][ind_output,:]
            if type == "why":
                sp_ind = np.where(cinfl_s < Cinfl_limit)
            elif type == "whynot":
                sp_ind = np.where(cinfl_s > Cinfl_limit)
            else:
                raise ValueError("Unknown explanation type")
            
        res_image = self.make_superpixels_transparent(sp_ind[0])
        return res_image
        
    def segmented(self, image):
        segments = slic(
            image,
            n_segments=self.nbr_segments,
            compactness=self.compactness,
            start_label=0,
        )
        return segments

    def perturbed_images(self, ind_inputs_to_explain, background_color):
        fudged_image = np.copy(self.image)
        for x in ind_inputs_to_explain:
              fudged_image[self.superpixels == x] = background_color
        return fudged_image

    def make_superpixels_transparent(self, sp_array):
        bg_colour = np.array(self.background_color)/255 if self.original_image.dtype == 'float32' else self.background_color
        fudged_image = np.copy(self.original_image)
        for x in sp_array:
            fudged_image[self.superpixels == x] = bg_colour
        return fudged_image


class SuperpixelPerturber:
    def __init__(self, segmenter, strategy, distortion):
        """
        @param segmenter: Callable that takes an image and produces an np.array of with segment indices for every pixel.
        @param strategy: Callable that takes an image and segmentation and returns a list order by segment containing lists of np.arrays indexing pixels to distorted.
        @param distortion: Callable that take an image and a index of pixels and applies a distortion to the indexed pixels.
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

        #Detect whether the image is in [0,1] or a  in [0,1,..,255]
        float_image = self.image.dtype == 'float32' #TODO: Make this detection more generic

        if not float_image:
            image = image/255

        #Segment the image
        superpixels = self.segmenter(image, **segmenter_kwds)

        #Create indices for where the images should be pertubed for each superpixel
        pertubation_masks = self.strategy(image, superpixels, **strategy_kwds)

        #Create pertubed images for each segment
        perturbed_images = []
        for segment_masks in pertubation_masks:
            segment_pertubed_images = []
            for image_mask in segment_masks:
                segment_pertubed_images.append(self.distortion(image, image_mask))
            perturbed_images.append(segment_pertubed_images)
        
        return (superpixels, perturbed_images)


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
        return slic(
        image,
        n_segments=self.nbr_segments,
        compactness=self.compactness,
        start_label=0,
    )


class EntireSegmentStrategy:
    def __init__(self, inverse=False):
        """
        @param bool inverse: Whether to mask every pixel except in the current segment or every pixel in the current segment
        """
        self.inverse = inverse
        self.a = 0 if inverse else 1
        self.b = 1 if inverse else 0

    def __call__(self, image, segments):
        indices = []
        segment_ids = np.unique(segments)
        for id in segment_ids:
            indices.append([np.where(segments==id,self.a,self.b)])
        return indices


class SingleColorDistortion:
    def __init__(self, background_color):
        """
        @param background_color: Background color to use for "transparent", in RGB. The default is gray (190, 190, 190).
        """
        self.background_color = np.array(background_color)
        self.background_color_float = self.background_color/255

    def __call__(self, image, distortion_mask):
        #Detect whether the image is in [0,1] or a  in [0,1,..,255]
        color = self.background_color_float if self.image.dtype == 'float32' else self.background_color #TODO: Make this detection more generic
        distorted_image = np.copy(image)
        distorted_image[distortion_mask] = color
        return distorted_image

        #distorted_images = []
        #for id_distortion_masks in distortion_masks:
        #    id_distorted_images = []
        #    for mask in id_distortion_masks:
        #        distorted_image = np.copy(image)
        #        distorted_image[mask] = color
        #        id_distorted_images.append(distorted_image)
        #    distorted_images.append(id_distorted_images)
        #return distorted_images

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