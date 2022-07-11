import numpy as np
import os 
from skimage import measure
from sklearn.cluster import KMeans
from skimage import morphology
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils.image_process import clip_and_normalize_mean_std, normalize_min_max_and_clip
from utils.image_io import load_ct_info
join = os.path.join
#from mayavi import mlab

             
def get_mask(npy_path, display=False):
    
    #读取原图
    img = np.load(npy_path)
    size = img.shape
    
    #K均值算法
    kmeans = KMeans(n_clusters=2).fit(np.reshape(img, [np.prod(img.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0) 
    
    #膨胀腐蚀
   # eroded = morphology.erosion(thresh_img, np.ones([2, 2, 2]))
   # dilation = morphology.dilation(eroded, np.ones([4, 4, 4]))
    
    #提取最大连通域
    label = measure.label(thresh_img, connectivity=2)
    props = measure.regionprops(label)
    numPix = []
    for ia in range(len(props)):
        numPix += [props[ia].area]    

    #选取目标连通域
    maxnum = sorted(numPix)[-2]
    #遍历每个连通区域
    for i in range(len(numPix)):
        #如果当前连通区域不是最大值所在的区域，则当前区域的值全部置为0，否则为1
        if numPix[i]!=maxnum:
            label[label==i+1]=0
        else:
            label[label==i+1]=1      
    
    mask = morphology.dilation(label, np.ones([5, 5, 5])) 

    lung = img*mask  
    
    if display:
        f, ax = plt.subplots(1,3,figsize=(10,10))
        ax[0].imshow(img[:,64,:],cmap='gray')
        ax[1].imshow(mask[:,64,:],cmap='gray')
        ax[2].imshow(lung[:,64,:],cmap='gray')
        plt.show()
    return mask, lung

if __name__ == '__main__':
    
    npy_dir = '/media/data/zhiqiang/npy/kt/'
    npy_list = sorted(os.listdir(npy_dir))
    save_dir = '/media/data/zhiqiang/mask/kt/'
    lung_save_dir = '/media/data/zhiqiang/lung/kt/'
    for i, npy_filename in enumerate(npy_list):
        if 'unuse' in npy_filename:
            continue
        npy_path = join(npy_dir, npy_filename)
        print(i, npy_filename)
        mask,lung = get_mask(npy_path, True)
        np.save(join(save_dir, npy_filename), mask)
        np.save(join(lung_save_dir, npy_filename), lung)