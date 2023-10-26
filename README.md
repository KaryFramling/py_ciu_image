# py.ciu.image
Contextual Importance and Utility (CIU) for explaining image classification

# Background

CIU was developed by Kary Främling in his PhD thesis *Learning and Explaining Preferences with Neural Networks for Multiple Criteria Decision Making*, (written in French, title *Modélisation et apprentissage des préférences par réseaux de neurones pour l'aide à la décision multicritère*), available online for instance here: https://tel.archives-ouvertes.fr/tel-00825854/document. It was originally implemented in Matlab and has later been re-implemented in R and Python (packages `ciu` and `py-ciu`) for tabular data. 

This `py.ciu.image` package implements CIU for explanation of image classification results. 

# Current status

This package was fundamentally modified in October 2023 to make it general-purpose (rather than being completely specific for the study of bleeding in gastro-enterological images). This work is still ongoing to some extent. The intention is to make it universally usable, like the initial R implementation ([https://github.com/KaryFramling/ciu.image](https://github.com/KaryFramling/ciu.image)) that was used in the paper. However, it is unlikely that the R package would be further developed due to the lack if deep neural network implementations in R and that the transfer of images between R and python is slow.

# Installation

The installation process is quite straightforward. Simply use `pip install` to obtain any of the below library imports that may be missing from your system, then create a CIU object following the parameter description. 

# Running

Check out the [CIU_gastro_images.ipynb](CIU_gastro_images.ipynb) notebook for examples of how to explain results of gastro-enterological images, as reported in the paper [Explainable artificial intelligence for human decision support system in the medical domain](https://www.mdpi.com/2504-4990/3/3/37).  

# Contributors

- [Kary Främling](https://github.com/KaryFramling): CIU "inventor", programmer of the initial [ciu.image code in "R"](https://github.com/KaryFramling/ciu.image)
- Vlad Apopei: Finalisation, programming and packaging of py.ciu.image.
- [Rohit Saluja](https://github.com/rohitsaluja1): Programming of the initial implementation of py.ciu.image.
- Manik Madhikermi: Support and testing. 
- Avleen Malhi: Initiator and co-ordinator of medical image explanation work. 
