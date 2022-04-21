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
        path,
        size,
        normalize,
        segments=None,
        compactness=None,
        strategy=None,
        CI=None,
        CU=None,
        debug=None,
    ):
        """
        @param model: TF/Keras model to be used.
        @param list out_names: List of output class names to be used.
        @param str path: The path of the image to be used.
        @param tuple size: The target size of the image.
        @param bool normalize: Whether the image input will be normalized to values between 0 an 1.
        @param int segments: The amount of target segments to be used by the SLIC algorithm. The default is 50.
        @param int compactness: The compactness of the segments accounting for proximity or RGB values. The default is 10
        and logarithmic.
        @param str strategy: Defines CIU strategy. Either "straight" or "inverse". The default is "inverse".
        @param float CI: Defines CI threshold to be used. The default is 0.01.
        @param float CU: Defines CU threshold to be used. The default is 0.1.
        @param bool debug: Displays variables for debugging purposes. The default is False.
        """

        self.model = model
        self.out_names = out_names
        self.path = path
        self.size = size
        self.normalize = normalize
        self.segments = segments if segments is not None else 50
        self.compactness = compactness if compactness is not None else 10
        self.strategy = strategy if strategy is not None else "inverse"
        self.CI = CI if CI is not None else 0.01
        self.CU = CU if CU is not None else 0.1
        self.debug = debug if debug is not None else False

    def image(self):
        img = cv2.imread(self.path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.size, 3)
        img = img.reshape(-1, self.size[0], self.size[1], 3)

        if self.debug:
            plt.imshow(img[0])
            plt.show()

        return img

    def segmented(self, image):
        segments = slic(
            image,
            n_segments=self.segments,
            compactness=self.compactness,
            max_iter=85,
            start_label=0,
        )

        if self.debug:
            for i in range(50):
                fig = plt.figure("Superpixels -- %d segments" % 50)
                ax = fig.add_subplot(1, 1, 1)
                ax.imshow(mark_boundaries(image[0], segments[0]))
                plt.axis("off")
            plt.show()

        return segments

    def ciu_image_result(self, ci, cu, cmin, cmax, outval, out_names):
        data = {
            "out_names": out_names,
            "CI": ci,
            "CU": cu,
            "cmin": cmin,
            "cmax": cmax,
            "outval": outval,
        }

        df = pd.DataFrame(data)
        return df

    def perturbed_images(self, image, ind_inputs_to_explain, segments):
        if self.normalize:
            img = image / 255

        fudged_image = img

        for x in ind_inputs_to_explain:
          if self.normalize:
              fudged_image[segments == x] = (190 / 255, 190 / 255, 190 / 255)
          else:
              fudged_image[segments == x] = (190, 190, 190)
          
        fudged_image_reshape = fudged_image.reshape(-1, self.size[0], self.size[1], 3)

        return fudged_image_reshape

    def explain_rgb(self, image, cu, pred):
        if self.normalize:
            img = image / 255

        cu_val = cu
        pert_val = pred

        if self.debug:
            print(f"CuVal: {cu_val}")
            print(f"Pert: {pert_val}")

        cu_val_flat = cu_val.ravel()
        pert_val_flat = pert_val.ravel()

        minvals = np.stack((cu_val_flat, pert_val_flat), axis=1).min(axis=1)
        maxvals = np.stack((cu_val_flat, pert_val_flat), axis=1).max(axis=1)

        diff = maxvals - minvals

        ci = diff

        ci_s = 0.5 if diff.any() == 0 else (cu_val_flat - minvals) / diff

        df_ciu_image_result = self.ciu_image_result(
            ci, ci_s, minvals, maxvals, cu_val_flat, self.out_names
        )

        sorted_df = df_ciu_image_result.sort_values(by=["outval"], ascending=False)

        return sorted_df

    def make_superpixels_transparent(self, image, sp_array, segments):
        fudged_image = image

        for x in sp_array:
            fudged_image[segments == x] = (190, 190, 190)

        if self.debug:
          plt.imshow(fudged_image[0])
          plt.title(f"Segments: {self.segments}, Compact: {self.compactness}, CI: {self.CI}")
          plt.show()

        return fudged_image[0]

    def CIU_Explanation(self):
        strategy = self.strategy
        image = self.image()
        ind_outputs = 1
        segments = self.segmented(image)
        all_sp = np.unique(segments)
        cu_base = (
            self.model.predict_on_batch(image / 255)
            if self.normalize
            else self.model.predict_on_batch(image)
        )

        if self.debug:
            print(all_sp)

        ci_s = np.zeros((ind_outputs, len(all_sp)))
        cu_s = np.zeros((ind_outputs, len(all_sp)))
        cmins = np.zeros((ind_outputs, len(all_sp)))
        cmaxs = np.zeros((ind_outputs, len(all_sp)))
        

        fudged_images = []

        for i in range(0, len(all_sp)):
          
            if strategy == "inverse":
                inps = np.delete(all_sp, i)
            elif strategy == "straight":
                inps = np.array([i])
            else:
                raise ValueError("Unknown Strategy")
            fudged_images.append(self.perturbed_images(image, inps, segments))

        predictions = self.model.predict_on_batch(np.vstack(fudged_images))

        for i in range(0, len(all_sp)):

            ciu = self.explain_rgb(image, cu_base, predictions[i])

            ci_s[:, i:i + 1] = ciu.iloc[0:ind_outputs, :]["CI"]
            cu_s[:, i:i + 1] = ciu.iloc[0:ind_outputs, :]["CU"]
            cmins[:, i:i + 1] = ciu.iloc[0:ind_outputs, :]["cmin"]
            cmaxs[:, i:i + 1] = ciu.iloc[0:ind_outputs, :]["cmax"]

        last_sp_ciu = {
            "out_names": ciu.iloc[0:ind_outputs, :]["out_names"],
            "out_vals": ciu.iloc[0:ind_outputs, :]["outval"],
            "CI_final": ci_s,
            "CU_final": cu_s,
            "cmin_final": cmins,
            "cmax_final": cmaxs,
        }

        last_sp_ciu["CU_final"] = np.nan_to_num(last_sp_ciu["CU_final"], nan=0.5)

        ci_s = last_sp_ciu["CI_final"]
        cu_s = last_sp_ciu["CU_final"]

        ci_s_inverse = 1 - ci_s

        sp_ind_df = pd.DataFrame(
            {"ci_s_inverse": ci_s_inverse.ravel(), "CUs": cu_s.ravel()},
            columns=["ci_s_inverse", "CUs"],
        )

        sp_ind = sp_ind_df.index[
            (sp_ind_df["ci_s_inverse"] < self.CI) | (sp_ind_df["CUs"] <= self.CU)
        ].tolist()

        if self.debug:
            print(f"IND_DF: {sp_ind_df}")
            print(sp_ind)

        image = self.make_superpixels_transparent(image, sp_ind, segments)

        return image
