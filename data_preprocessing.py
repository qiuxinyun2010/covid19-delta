import os
from glob import glob
from utils.image_io import load_ct_info
from utils.image_process import change_axes_of_image
from utils.image_resample import ScipyResample
import torchio as tio
import matplotlib.pylab as plt
import SimpleITK as sitk
import torchio as tio
import pydicom
import numpy as np

join = os.path.join



# *.dcm => *.nii.gz
def get_nii(dcm_path="/media/data/zhiqiang/dcm/delta",nii_path="/media/data/zhiqiang/nii/delta"):
    for idx, dcm_filename in enumerate(os.listdir(dcm_path)):
        print(idx, dcm_filename)
        study_name = dcm_filename
        dcm_filename = join(dcm_path, dcm_filename)
        dicom_names = {}
        p = None
        for i, dcm in enumerate(sorted(os.listdir(dcm_filename))):
            #print(i, join(dcm_filename, dcm))
            p = pydicom.read_file(join(dcm_filename, dcm))
            if  not hasattr(p,'SliceLocation'):
                continue    
            dicom_names[join(dcm_filename, dcm)] = abs(p.SliceLocation)
            
        #根据切片位置，对文件名进行排序
        sort = sorted(dicom_names.items(), key=lambda d: d[1])
        dicom_names_sort = [s[0] for s in sort]

        #组合切片，变成np array
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_names_sort)
        image2 = reader.Execute()
        image_array = sitk.GetArrayFromImage(image2) 
        origin = image2.GetOrigin()  # x, y, z
        spacing = image2.GetSpacing()  # x, y, z
        direction = image2.GetDirection()  # x, y, z

        # 3.将array转为img，并保存为.nii.gz
        image3 = sitk.GetImageFromArray(image_array)
        image3.SetSpacing(spacing)
        image3.SetDirection(direction)
        image3.SetOrigin(origin)
        sitk.WriteImage(image3, join(nii_path,manufacturer,study_name+'.nii'))    
        x1 = tio.ScalarImage(join(nii_path,manufacturer,study_name+'.nii'))
        x1.plot()
       # print(x1,manufacturer)

# *.nii.gz => *.npy
def get_npy(nii_dir, save_dir, out_put_size=(128,128,128), display=False):
    nii_list = sorted(os.listdir(nii_dir))
    for i, nii_filename in enumerate(nii_list):
        image_info = load_ct_info(join(nii_dir, nii_filename))
        
        npy_image = image_info['npy_image']
        image_direction = image_info['direction']
        image_spacing = image_info['spacing']        
        print(i, nii_filename,  "Size:",npy_image.shape, "image_direction:", image_direction, "image_spacing:", image_spacing)
        npy_image = change_axes_of_image(npy_image, image_direction)
        size = npy_image.shape
        out_put_size =[]
        for _size in size:
            if _size > 256:
                out_put_size.append(int(_size/2))
            else:out_put_size.append(_size)
        out_put_size = tuple(out_put_size)
       # print(out_put_size)
        resample_image, _ = ScipyResample.resample_to_size(npy_image, out_put_size)
        print("original_size", size, "after_size", resample_image.shape)
        if display:
            plt.imshow(resample_image[:,int(out_put_size[1]/2),:], cmap='gray')
            plt.show()
        np.save(join(save_dir, nii_filename+'.npy'), resample_image)