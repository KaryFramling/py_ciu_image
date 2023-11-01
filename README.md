# py.ciu.image
Contextual Importance and Utility (CIU) for explaining image classification

## Background

CIU was developed by Kary Främling in his PhD thesis *Learning and Explaining Preferences with Neural Networks for Multiple Criteria Decision Making*, (written in French, title *Modélisation et apprentissage des préférences par réseaux de neurones pour l'aide à la décision multicritère*), available online for instance here: https://tel.archives-ouvertes.fr/tel-00825854/document. It was originally implemented in Matlab and has later been re-implemented in R and Python (packages `ciu` and `py-ciu`) for tabular data. 

This `py.ciu.image` package implements CIU for explanation of image classification results. 

## Current status

This package was fundamentally modified in October 2023 to make it general-purpose (rather than being completely specific for the study of bleeding in gastro-enterological images). 


## Installation

For the moment, there's no PyPI package of `py.ciu.image` so the best option is probably to clone this repository.  

A Jupyter notebook is provided that shows how to produce CIU explanations for Imagenet networks (VGG16 and Resnet-152 are used). The notebook is [CIU_Imagenet.ipynb](CIU_Imagenet.ipynb).

Another Jupyter notebook goes through how gastroenterological images can be "explained" with CIU for bleeding and non-bleeding images, as reported in the paper [Explainable artificial intelligence for human decision support system in the medical domain](https://www.mdpi.com/2504-4990/3/3/37). The notebook is [CIU_gastro_images.ipynb](CIU_gastro_images.ipynb). The original 3295 images in the Red Lesion Endoscopy data set are publicly available and can be retrieved from Coelho [https://rdm.inesctec.pt/dataset/nis-2018-003](https://rdm.inesctec.pt/dataset/nis-2018-003), last accessed on 26 October 2023.

## Related resources

The initial R implementation is accessible at ([https://github.com/KaryFramling/ciu.image](https://github.com/KaryFramling/ciu.image)) and was also used in the gastroenterological paper from 2021. However, it is unlikely that the R package would be further developed due to the lack if deep neural network implementations in R and that the transfer of images between R and python is slow.

## Contributors

- [Kary Främling](https://github.com/KaryFramling): CIU "inventor", programmer of the initial [ciu.image code in "R"](https://github.com/KaryFramling/ciu.image) and of the total re-factoring and fixing of the package in Oct/Nov 2023.
- Vlad Apopei: Finalisation, programming and packaging of py.ciu.image.
- [Rohit Saluja](https://github.com/rohitsaluja1): Programming of the initial implementation of py.ciu.image.
- Manik Madhikermi: Support and testing. 
- Avleen Malhi: Initiator and co-ordinator of medical image explanation work. 
