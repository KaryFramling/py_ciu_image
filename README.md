# py.ciu.image
Contextual Importance and Utility (CIU) for explaining image classification

# Background

CIU was developed by Kary Främling in his PhD thesis *Learning and Explaining Preferences with Neural Networks for Multiple Criteria Decision Making*, (written in French, title *Modélisation et apprentissage des préférences par réseaux de neurones pour l'aide à la décision multicritère*), available online for instance here: https://tel.archives-ouvertes.fr/tel-00825854/document. It was originally implemented in Matlab and has later been re-implemented in Python and R (package `ciu`) for tabular data. 

This `py.ciu.image` package implements CIU for explanation of image classification results. 

# Current status

This package will be fundamentally modified "soon". It is now almost completely specific for the study of bleeding in gastro-enterological images, reported in the paper [Explainable artificial intelligence for human decision support system in the medical domain](https://www.mdpi.com/2504-4990/3/3/37). The intention is to make it universally usable, like the initial R implementation ([https://github.com/KaryFramling/ciu.image](https://github.com/KaryFramling/ciu.image)) that was used in the paper. However, it is unlikely that the R package would be further developed due to the lack if deep neural network implementations in R and that the transfer of images between R and python is slow.

# Installation

The installation process is quite straightforward. Simply use `pip install` to obtain any of the below library imports that may be missing from your system, then create a CIU object following the parameter description. 

# Running

Check out the [README_notebook.ipynb](README_notebook.ipynb) notebook for the most recent examples and for running it yourself. 


Below a simple example is shown using an endoscopy image. 

``` python
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from CIU import CIU

model = tf.keras.models.load_model("model_full_categorical.h5")
image_path = "Set1_674.png"
out_names = ["NonBleeding", "Bleeding"]

ciu_object = CIU(model, out_names, image_path, (150, 150), True)

tic = time.perf_counter()
image_output = ciu_object.CIU_Explanation()
toc = time.perf_counter()
print(f"Done in {toc - tic:0.4f} seconds")

plt.imshow(image_output)
plt.show()
```

# Contributors

- [Kary Främling](https://github.com/KaryFramling): CIU "inventor", programmer of the initial [ciu.image code in "R"](https://github.com/KaryFramling/ciu.image)
- Vlad Apopei: Finalisation, programming and packaging of py.ciu.image.
- [Rohit Saluja](https://github.com/rohitsaluja1): Programming of the initial implementation of py.ciu.image.
- Manik Madhikermi: Support and testing. 
- Avleen Malhi: Initiator and co-ordinator of medical image explanation work. 
