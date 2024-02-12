import os
import glob
import pickle
import pandas as pd
import numpy as np
import SimpleITK as sitk
from lungmask import mask

CT_DIR = '../data/ct_raw/'
DATA_DIR = '../data/'
SEG_DATA_DIR = '../data/data_seg/'


def main():
    paths = sorted(glob.glob(os.path.join(CT_DIR, '*.nii')))

    for path in paths:
        file_name = os.path.basename(path).split('.nii')[0]
        ct = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(ct)

        segmentation = mask.apply(ct)  # R231 mask
        segmentation[segmentation != 0] = 1

        masked = np.copy(img)
        masked[segmentation == 0] = 1024  # maximum 1024, minimum -1024
        masked = masked / 2048 + 0.5  # normalize between 0 and 1
        masked = masked.astype(np.float32)

        # remove empty regions in axial coordiante
        lung_region = np.mean(segmentation, axis=(1, 2))
        start = np.where(lung_region > 0)[0][0]
        end = np.where(lung_region > 0)[0][-1]
        num_slice = end - start+ 1
        masked = masked[start:end+1]

        # add label
        clinic = pd.read_csv(os.path.join(DATA_DIR + 'ct_names.csv'))

        if any(clinic['ID'] == file_name):
            label = clinic[clinic['ID'] == file_name].values[0][1:3].astype('int') 
            clinic_feature = clinic[clinic['ID'] == file_name].values[0][3:].astype('int')

            with open(os.path.join(SEG_DATA_DIR, f'seg_{file_name}_{num_slice}_{label[0]}_{label[1]}.pickle'), 'wb') as handle:
                pickle.dump(masked, handle)
                pickle.dump(num_slice, handle)
                pickle.dump(label, handle)
                pickle.dump(clinic_feature, handle)


if __name__ == "__main__":
    main()
