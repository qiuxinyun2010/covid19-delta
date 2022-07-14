# use 3D-CNN to identify delta variant of covid-19 based on Chest CT

## Pipeline
![Pipeline](./src/QIMS-22-193-R1-FIG1-5070.png)

## Data preprocessing 
* Convert CT image from the DICOM format to the NIfTI format. (*.dcm => *.nii.gz)
* Images are reoriented to the same direction and resampled to the fixed size (128×128×128).  (*.nii.gz = > *.npy)
* Use K-means algorithm to extract lung region from CT images. 
* The intensity values are clipped to the range [-1024, 1024] and a z-score normalization is applied based on the mean and standard deviation of the intensity values. 
## Method
![Method](./src/QIMS-22-193-R1-FIG3-2546.png)