"""
This package implements the Contextual Importance and Utility (CIU) method.

Classes: 
    - :class:`CIUimage.CIUimage`: The CIUimage class implements the Contextual Importance and Utility method for explaining image classification. 

Example:
::

    # Example code using the module
    import CIUimage as CIU
    out_names = ["NonBleeding", "Bleeding"] # Can also be "None".
    ciu_object = CIU.CIUimage(model, out_names)
    ciu_sp_result = ciu_object.explain(img_to_xplain)
"""
from .CIUimage import CIUimage

