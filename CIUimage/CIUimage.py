from typing import Any
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from skimage.filters import gaussian
import pandas as pd
import cv2
import numbers
import numpy as np
np.seterr(divide="ignore", invalid="ignore")


class CIUimage:
    def __init__(
        self,
        model,
        out_names,
        predict_function=None,
        perturber = None,
        neutralCU = 0.5,
        debug = False,
        inverse = False
    ):
        """
        This class implements the Contextual Importance and Utility (CIU) explainable AI method for explaining 
        image classifications.

        @param model: TF/Keras model to be used.
        @param list out_names: List of output class names to be used.
        @param predict.function: Function that takes a list of images and return a numpy.ndarray with output probabilities. 
        @param perturber: Callable that takes images and returns a tuple of (superpixels, segment masks, pertubation masks, perturbed images).
            superpixels is a list of ndarrays for each image of shape [H,W] indicating which superpixel that pixel belongs to (deprecated).
            segement masks is a list of ndarrays for each image of shape [S,H,W] masking each pixel as 0 or 1 depending on if it belongs to each segment s.
            pertubation masks is a list of ndarrays of shape [S,E,H,W] with E masks for each segment indicating which pixels should be perturbed.
            perturbed images is a list of ndarrays for each image of shape [S,P,H,W,C] of P perturbed images for each segment s in S.
        image and returns a list of segmentation masks and a list of lists with
        distorted images for each segmentation mask. The default is to use SlicOcclusionPerturber.
        @param neutralCU: CU value that is considered "neutral" and that provides a limit between negative and 
        positive influence in the "Contextual influence" calculation CIx(CU - neutralCU).
        @param bool debug: Displays variables for debugging purposes. The default is False.
        @param bool inverse: Whether to calculate ci as 1-ci, useful when a segment is perturbed by altering other segments.
        """

        self.model = model
        self.out_names = out_names
        self.predict_function = predict_function if predict_function is not None else model.predict_on_batch
        self.perturber = perturber
        self.neutralCU = neutralCU
        self.debug = debug
        self.inverse = inverse

        if self.perturber is None:
            strategy = "inverse" if inverse else "straight"
            self.perturber = SlicOcclusionPerturber(strategy=strategy)

        # Set internal object variables to default values.
        self.original_image = None
        self.image = None
        self.superpixels = None #TODO: Currenlty using superpixels despite being redundant and not as general
        self.segments = None
        self.masks = None
        self.ciu_segments = None

    # The method to call for getting CIU values for all superpixels. It returns a 'dict' object. 
    def explain(self, image):
        """
        Calculate CIU values for the given image. 

        @param image: The image to be explained.
        """ 
        
        # Memorize image, store shape also
        self.original_image = image
        self.image = np.copy(self.original_image)
        self.image_shape = self.image.shape
        
        # Necessary reshape for model predict, not so nice to have that here, 
        # should still be abstracted away somehow.
        dnn_img = self.image.reshape(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])
        
        superpixels, segment_masks, pertubation_masks, perturbed_images = self.perturber(dnn_img)
        superpixels = superpixels[0] #TODO: Hardcoded to work with 1 image for now
        segment_masks = segment_masks[0]
        perturbed_images = perturbed_images[0]
        nbr_segments = len(segment_masks)
        self.segments = segment_masks
        self.masks = pertubation_masks
        self.superpixels = superpixels

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
            if self.inverse:
                ci = 1 - ci
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
    def image_influential_segments_only(self, ind_output=0, Cinfl_limit=None, type = "why", CI_limit=0.5, CU_limit=0.51, use_perturber=False):
        # Do based on CI and CU
        influential_segments = self.get_influential_segments(ind_output, Cinfl_limit, type, CI_limit, CU_limit)
        unfluential_segments = [np.where(s==0, 1, 0) for s in influential_segments]
        if use_perturber:
            distortion = self.perturber.distortion
        else:
            distortion = SingleColorDistortion((190,190,190))
        return np.squeeze(distortion(np.expand_dims(self.image, axis=0), unfluential_segments), axis=(0,1))[0]

    
    def get_influential_segments(self, ind_output=0, Cinfl_limit=None, type = "why", CI_limit=0.5, CU_limit=0.51):
        # Do based on CI and CU
        if Cinfl_limit is None:
            ci_s = self.ciu_segments["CI"][ind_output,:]
            cu_s = self.ciu_segments["CU"][ind_output,:]
            segments = np.where(np.logical_not((ci_s < CI_limit) | (cu_s < CU_limit)))
        else:
            cinfl_s = self.ciu_segments["Cinfl"][ind_output,:]
            if type == "why":
                segments = np.where(np.logical_not(cinfl_s < Cinfl_limit))
            elif type == "whynot":
                segments = np.where(np.logical_not(cinfl_s > Cinfl_limit))
            else:
                raise ValueError("Unknown explanation type")
        
        influential_segments=np.zeros(self.segments[0].shape)
        for id, segment in enumerate(self.segments[segments[0]]):
            if id == 0:
                influential_segments = segment
            else:
                influential_segments = np.where(segment==0, influential_segments, segment)
        return [np.expand_dims(influential_segments, axis=(0,1))]

class SuperpixelPerturber:
    def __init__(self, segmenter, strategy, distortion):
        """
        A wrapper class to create perturbers for CIUimage that generates perturbed images used in CIU calcualtion.

        @param segmenter: Callable that takes images and produces a (superpixels, segment masks) tuple.
            superpixels is a list of [H,W] ndarrays for each image indicating which superpixel each pixel belongs to.
            segment masks is a list of [S,H,W] np.arrays for each image masking each segment in S with 0 and 1.
        @param strategy: Callable that takes images and segment masks and returns pertubation masks.
            pertubation masks is a list of [S,E,H,W] ndarrays for each image with E pertubation masks for each segment. 
        @param distortion: Callable that takes images and pertubation masks and returns perturbed images.
            perturbed images is a list of [S,P,H,W,C] ndarrays for each image containing P perturbed images for each segment.
        """
        
        self.segmenter = segmenter
        self.strategy = strategy
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
        superpixels, segment_masks = self.segmenter(image, **segmenter_kwds)

        #Create indices for where the images should be pertubed for each superpixel
        pertubation_masks = self.strategy(image, segment_masks, **strategy_kwds)

        #Create pertubed images for each segment
        perturbed_images = self.distortion(image, pertubation_masks, **distortion_kwds)
        
        return (superpixels, segment_masks, pertubation_masks, perturbed_images)

class SlicSegmenter:
    def __init__(self, nbr_segments, compactness):
        """
        A segmenter that produces SLIC superpixels and segmentation masks when called with input images.

        @param int nbr_segments: The amount of target segments to be used by the SLIC algorithm. The default is 50.
        @param int compactness: The compactness of the segments accounting for proximity or RGB values. The default is 10
        and logarithmic.
        """
        self.nbr_segments = nbr_segments
        self.compactness = compactness

    def __call__(self, image):
        """
        @param image: Image to segment.
        """
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
                image_masks.append(np.where(image_segments==id,1.0,0.0))
            segment_masks.append(np.array(image_masks))
        return segments, segment_masks

class GridSegmenter:
    def __init__(self, size):
        """
        @param int size:
        """
        self.nbr_size = size

    #TODO: Implement

class EntireSegmentStrategy:
    def __init__(self, inverse=False, fade_fun=None, **kwargs):
        """
        A strategy that produces a list of [S,1,H,W] ndarrays for each image containing pertubations masks for each segmentation s.
        The pertubation masks are 1 for the pixels to be perturbed and 0 for those that don't.
        The fade_fun parameter can cause values to be between 0 and 1 indicating weaker pertubation.

        @param bool inverse: Whether to mask every pixel except in the current segment or every pixel in the current segment.
        @param fade_fun: Image transform applied to each segment to fade the segment border between 0 and 1.
        @param **kwargs: Key word arguments used by fade_fun.
        """
        self.inverse = inverse
        self.fade_fun = fade_fun
        self.kwargs = kwargs

    def __call__(self, image, segment_masks):
        """
        @param image: Unused. Needed by strategies used by SuperpixelPerturber.
        @param segment_masks: A list of [S,H,W] np.arrays for each image masking each segment in S with 0 and 1.
        """
        distortion_masks = []
        if self.inverse:
            for image_masks in segment_masks:
                distortion_masks.append(np.expand_dims(np.where(image_masks==0,1.0,0.0),axis=1))
        else: 
            for image_masks in segment_masks:
                distortion_masks.append(np.expand_dims(image_masks,axis=1))

        if not self.fade_fun is None:
            for image_masks in distortion_masks:
                for mask in image_masks:
                    mask[:] = self.fade_fun(mask, **self.kwargs)
        return distortion_masks

class SingleColorDistortion:
    def __init__(self, background_color):
        """
        A distorter that produces a list of [S,P,H,W,C] ndarrays for each image containing perturbed versions of the images.
        The perturbed images consist of the pixels indicated by the [S,P,H,W] pertubation mask being replaced by a static color.

        @param background_color: Background color to use for "transparent", in RGB. The default is gray (190, 190, 190).
        """
        self.background_color = np.array(background_color)
        self.background_color_float = self.background_color/255

    def __call__(self, image, distortion_masks):
        """
        @param image: The images to perturb.
        @param distortion_masks: A list of [S,P,H,W] ndarrays for each image with P pertubation masks for each segment. 
        """
        #Detect whether the image is in [0,1] or a  in [0,1,..,255]
        color = self.background_color if issubclass(image.dtype.type, numbers.Integral) else self.background_color_float

        distorted_images = []
        for id, image_distortion_masks in enumerate(distortion_masks):
            indices = np.nonzero(image_distortion_masks)
            distorted_image = np.tile(np.copy(image[id]), (image_distortion_masks.shape[0],image_distortion_masks.shape[1],1,1,1))
            distorted_image[indices] = color
            distorted_images.append(distorted_image)
        return distorted_images

class GaussianBlurDistortion:
    def __init__(self, sigma, truncate=None, ksize=None):
        """
        A distorter that produces a list of [S,P,H,W,C] ndarrays for each image containing perturbed versions of the images.
        The perturbed images consist the pixels indicated by the [S,P,H,W] pertubation mask being blurred with a Gaussian kernel.

        @param float sigma: Std. dev. used by the Gaussian kernel used for blurring
        @param float truncate: If set, truncates the Gaussian kernel after that many std. devs. (cannot be used with ksize)
        @param int ksize: If set, truncates the Gaussian kernel to size (ksize, ksize) (cannot be used with truncate)
        """
        self.sigma = sigma
        if not (truncate is None or ksize is None):
            raise ValueError("Parameters truncate and ksize cannot both be set")
        if truncate is None:
            self.truncate = (((ksize-1)/2)-0.5)/sigma
        else:
            self.truncate = truncate

    def __call__(self, image, distortion_masks):
        """
        @param image: The images to perturb.
        @param distortion_masks: A list of [S,P,H,W] ndarrays for each image with P pertubation masks for each segment. 
        """
        distorted_images = []
        for id, image_distortion_masks in enumerate(distortion_masks):
            indices = np.nonzero(image_distortion_masks)
            blurred_image = gaussian(np.copy(image[id]), sigma=self.sigma, truncate=self.truncate)
            blurred_image = np.tile(blurred_image, (image_distortion_masks.shape[0],image_distortion_masks.shape[1],1,1,1))
            distorted_image = np.tile(np.copy(image[id]), (image_distortion_masks.shape[0],image_distortion_masks.shape[1],1,1,1))
            distorted_image[indices] = blurred_image[indices]
            distorted_images.append(distorted_image)
        return distorted_images

class TransformDistortion:
    def __init__(self, transform_fun, **kwargs):
        """
        A distorter that produces a list of [S,P,H,W,C] ndarrays for each image containing perturbed versions of the images.
        The perturbed images consist of the pixels indicated by the [S,P,H,W] pertubation mask being transformed.
        
        @param transform_fun: A function that takes an image as 1st param and applies a transform to it
        @param **kwargs: All other params passed to transform_fun as keyword arguments
        """
        self.transform_fun = transform_fun
        self.kwargs = kwargs

    def __call__(self, image, distortion_masks):
        """
        @param image: The images to perturb.
        @param distortion_masks: A list of [S,P,H,W] ndarrays for each image with P pertubation masks for each segment. 
        """
        distorted_images = []
        for id, image_distortion_masks in enumerate(distortion_masks):
            indices = np.nonzero(image_distortion_masks)
            transformed_image = self.transform_fun(np.copy(image[id]), **self.kwargs)
            transformed_image = np.tile(transformed_image, (image_distortion_masks.shape[0],image_distortion_masks.shape[1],1,1,1))
            distorted_image = np.tile(np.copy(image[id]), (image_distortion_masks.shape[0],image_distortion_masks.shape[1],1,1,1))
            distorted_image[indices] = transformed_image[indices]
            distorted_images.append(distorted_image)
        return distorted_images
    
class ReplaceImageDistortion:
    def __init__(self, default_images=None):
        """
        A distorter that produces a list of [S,P,H,W,C] ndarrays for each image containing perturbed versions of the images.
        The perturbed images consist of the pixels indicated by the [S,E,H,W] pertubation mask being replaced by pixels in the indicated images.
        The number of perturbed images per segment, P, is the number of pertubation masks per segment, E, times the number of replacement images.

        @param default_images: The image(s) to be replace the indicated segments of the image
        """
        self.default_images = default_images

    def __call__(self, image, distortion_masks):
        """
        @param image: The images to perturb.
        @param distortion_masks: A list of [S,E,H,W] ndarrays for each image with P pertubation masks for each segment. 
        """
        distorted_images = []
        for id, image_distortion_masks in enumerate(distortion_masks):
            replacement_images = np.tile(np.copy(self.default_images), (image_distortion_masks.shape[0],image_distortion_masks.shape[1],1,1,1))
            image_distortion_masks = np.tile(image_distortion_masks, (1,self.default_images.shape[0],1,1))
            indices = np.nonzero(image_distortion_masks)
            distorted_image = np.tile(np.copy(image[id]), (image_distortion_masks.shape[0],image_distortion_masks.shape[1],1,1,1))
            distorted_image[indices] = replacement_images[indices]
            distorted_images.append(distorted_image)
        return distorted_images

def SlicOcclusionPerturber(background_color=(190,190,190), strategy="straight", nbr_segments=50, compactness=10):
    """
    Returns a instantiated SuperpixelPerturber that replicates the behavior of the original implementation of CIUimage pertubations.

    @param background_color: Background color to use for "transparent", in RGB. The default is gray (190, 190, 190). 
    @param bool strategy: Defines whether the CIU strategy should be "inverse" or "straight". The default is "straight".
    @param int nbr_segments: The amount of target segments to be used by the SLIC algorithm. The default is 50.
    @param int compactness: The compactness of the segments accounting for proximity or RGB values. The default is 10 and logarithmic.
    """
    segmenter = SlicSegmenter(nbr_segments, compactness)
    strategy = EntireSegmentStrategy(inverse=strategy=="inverse")
    distortion = SingleColorDistortion(background_color)
    return SuperpixelPerturber(segmenter, strategy, distortion)
