# py.ciu.image
Contextual Importance and Utility (CIU) for image explanation

# Background

CIU was developed by Kary Främling in his PhD thesis *Learning and Explaining Preferences with Neural Networks for Multiple Criteria Decision Making*, (written in French, title *Modélisation et apprentissage des préférences par réseaux de neurones pour l'aide à la décision multicritère*), available online for instance here: https://tel.archives-ouvertes.fr/tel-00825854/document. It was originally implemented in Matlab and has later been re-implemented in Python and R (package `ciu`) for tabular data. 

This `ciu.image` package implements CIU for image recognition "explanation" using e.g. saliency maps. 

# Installation

The installation process is quite straightforward. Simply use `pip install` to obtain any of the below library imports that may be missing from your system, then create a CIU object following the parameter description. 

# Running

Below a simple example is shown using an endoscopy image. 

``` python
import time
import matplotlib.pyplot as plt

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
- Rohit Saluja: Initial implementation of py.ciu.image.
- Manik Madhikermi: Support and testing. 
- Avleen Malhi: Initiator and co-ordinator of medical image explanation work. 
