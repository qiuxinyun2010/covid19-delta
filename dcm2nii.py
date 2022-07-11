import SimpleITK as sitk
import os
import torchio as tio
import pydicom
import numpy as np

join = os.path.join
dcm_path = "/media/data/zhiqiang/dcm/delta"
nii_path = "/media/data/zhiqiang/nii/delta"
hu_min = []
direction = []
size = []

if __name__ == '__main__':
    #遍历所有dcm切片，记录每个切片的位置信息SliceLocation
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
        print(x1,manufacturer)