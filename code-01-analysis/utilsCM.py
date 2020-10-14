import os
from PIL import Image
import numpy as np
import scipy.io
import torch
from torchvision.transforms import functional as tr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats



#Modified by Cate 11 August 2020, Subject function to make it fit with local directory folder name

# ImageNet mean and standard deviation. All images
# passed to a PyTorch pre-trained model (e.g. AlexNet) must be
# normalized by these quantities, because that is how they were trained.
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


def listdir(dir, path=True):
    files = os.listdir(dir)
    files = [f for f in files if f != '.DS_Store']
    files = sorted(files)
    if path:
        files = [os.path.join(dir, f) for f in files]
    return files


def image_to_tensor(image, resolution=None, do_imagenet_norm=True):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    if resolution is not None:
        image = tr.resize(image, resolution)
    if image.width != image.height:     # if not square image, crop the long side's edges
        r = min(image.width, image.height)
        image = tr.center_crop(image, (r, r))
    image = tr.to_tensor(image)
    if do_imagenet_norm:
        image = imagenet_norm(image)
    return image


def tensor_to_image(image, do_imagenet_unnorm=True):
    if do_imagenet_unnorm:
        image = imagenet_unnorm(image)
    image = tr.to_pil_image(image)
    return image


def imagenet_norm(image):
    dims = len(image.shape)
    if dims < 4:
        image = [image]
    image = [tr.normalize(img, mean=imagenet_mean, std=imagenet_std) for img in image]
    image = torch.stack(image, dim=0)
    if dims < 4:
        image = image.squeeze(0)
    return image


def imagenet_unnorm(image):
    mean = torch.tensor(imagenet_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(imagenet_std, dtype=torch.float32).view(3, 1, 1)
    image = image.cpu()
    image = image * std + mean
    return image


class Subject:

    def __init__(self, subject_num, rois):
        roistack = scipy.io.loadmat('data-object2vec/rois043/subj{:03}'.format(subject_num) +
                                    '/roistack.mat')['roistack']

        self.roi_names = [d[0] for d in roistack['rois'][0, 0][:, 0]]
        self.conditions = [d[0] for d in roistack['conds'][0, 0][:, 0]]

        roi_indices = roistack['indices'][0, 0][0]
        roi_masks = {roi: roi_indices == (i + 1) for i, roi in enumerate(self.roi_names)}
        voxels = roistack['betas'][0, 0]
        self.condition_voxels = {cond: np.concatenate([voxels[i][roi_masks[r]] for r in rois])
                                 for i, cond in enumerate(self.conditions)}
        self.n_voxels = np.sum([roi_masks[r] for r in rois])

        sets = scipy.io.loadmat('data-object2vec/rois043/subj{:03}'.format(subject_num) +
                                   '/sets.mat')['sets']
        self.cv_sets = [[cond[0] for cond in s[:, 0]] for s in sets[0, :]]


def cv_regression(condition_features, subject, l2=0.0):
    # Get cross-validated mean test set correlation
    rs = []
    for test_conditions in subject.cv_sets:
        train_conditions = [c for c in subject.conditions if c not in test_conditions]
        train_features = np.stack([condition_features[c] for c in train_conditions])
        test_features = np.stack([condition_features[c] for c in test_conditions])
        train_voxels = np.stack([subject.condition_voxels[c] for c in train_conditions])
        test_voxels = np.stack([subject.condition_voxels[c] for c in test_conditions])
        _, r = regression(train_features, train_voxels, test_features, test_voxels, l2=l2)
        rs.append(r)
    mean_r = np.mean(rs)

    # Train on all of the data
    train_conditions = subject.conditions
    train_features = np.stack([condition_features[c] for c in train_conditions])
    train_voxels = np.stack([subject.condition_voxels[c] for c in train_conditions])
    weights = regression(train_features, train_voxels, None, None, l2=l2, validate=False)

    return weights, mean_r


def regression(x_train, y_train, x_test, y_test, l2=0.0, validate=True):
    regr = Ridge(alpha=l2, fit_intercept=False)
    regr.fit(x_train, y_train)
    weights = regr.coef_
    if validate:
        y_pred = regr.predict(x_test)
        r = correlation(y_test, y_pred)
        return weights, r
    else:
        return weights


def correlation(a, b):
    np.seterr(divide='ignore', invalid='ignore')
    zs = lambda v: (v - v.mean(0)) / v.std(0)
    r = (zs(a) * zs(b)).mean()
    return r

def iter_cvregress(X_features,Y,keyword,ilayer,pc = None,iROI = [],k=9,savefolder = None, Ypredict = None, pretrained = True):
    #How many fold to compute?
    kf = KFold(n_splits = k)
    l2 = 0.0
    if pc is not None:
        pca = PCA(n_components=pc)

    if iROI:
        print('k-fold regression, independet variable: ' + str(pc) + ' PCs retained of ' + keyword + ' from ' + iROI)
        # filename = 'PredictSENSES_' + keyword + '_' +iROI + '_'+ ilayer + '_'+ str(pc) +'PCs'
    else:
        print('k-fold regression, independet variable: ' + str(pc) + ' PCs retained of ' + keyword + ' from ' + ilayer)
        # filename = 'PredictSENSES_' + keyword + '_'+ ilayer + '_'+ str(pc) +'PCs'
        
    rs = []#[[] for i in (0,(Y.values.shape[1]-1))]
    
    for train_index, test_index in kf.split(X_features):
        train_features = X_features[train_index,]
        test_features = X_features[test_index,]
        if pc is not None:
            pca.fit(X_features[train_index,])
            train_features = pca.transform(train_features)
            test_features = pca.transform(test_features)
        train_Y = Y[train_index,]
        test_Y = Y[test_index,]
        
        r = []
        _,r = regression_iter(train_features, train_Y, test_features,  test_Y, l2=l2)
        rs.append(r)

    rs = np.array(rs)

    # print('r: ' , len(r))
    # print('rs: ' , rs.shape)

    mean_r = np.nanmean(rs, axis=0) #TO handle Nans, since the feature space is so sparse
    # print(mean_r)
    
    #Saved information
    if savefolder is not None:
        if iROI:
            filename = 'Predict' + Ypredict + '_' + keyword + '_' +iROI + '_'+ ilayer + '_'+ str(pc) +'PCs'
        else:
            filename = 'Predict' + Ypredict + '_' + keyword + '_'+ ilayer + '_'+ str(pc) +'PCs'
        if not pretrained:
            np.save(savefolder + filename + '_untrained', mean_r)
        else:
            np.save(savefolder + filename, mean_r)
            
    return mean_r



def regression_iter(x_train, y_train, x_test, y_test, l2=0.0, validate=True):
    regr = Ridge(alpha=l2, fit_intercept=False)
    regr.fit(x_train, y_train)
    weights = regr.coef_
#     print(weights[0,0:3])
    r_ = []

    if validate:
        y_pred = regr.predict(x_test)
        for (y_t, y_p) in zip(y_test.transpose(), y_pred.transpose()):
            r = correlation(y_t, y_p)
            r_.append(r)
        return weights,r_
    else:
        return weights
 
def make_figure(mean_r,keyword,layer,icomps,iROI = 'Dnn',figure_size=(100,35),figure_path = None, Ypredict = None, font_size = 100,pretrained = True) :
    if iROI is not 'Dnn':
        title =  'Predict' + Ypredict + '_' + keyword + ' ' + iROI + ' ' + layer + ' '+ str(icomps) +'PCs'
        filename =  'Predict' + Ypredict + '_' + keyword + '_' +iROI + '_'+ layer + '_'+ str(icomps) +'PCs'
    else:
        title =  'Predict' + Ypredict + '_' + keyword + ' ' + layer + ' '+ str(icomps) +'PCs'
        filename =  'Predict' + Ypredict + '_' + keyword + '_'+ layer + '_'+ str(icomps) +'PCs'
        
    plt.figure(figsize = figure_size)
    # Set the font dictionaries (for plot title and axis titles)
    title_font = {'fontname':'Arial', 'size':str(font_size), 'color':'black', 'weight':'normal',
  'verticalalignment':'bottom'} # Bottom vertical alignment for more space
    axis_font = {'fontname':'Arial', 'size':str(font_size)}
    sns.set()
    # n= len(~np.isnan(mean_r))
    mean_r_nonan = mean_r[~np.isnan(mean_r)]
    n= len(mean_r_nonan)
    ax = plt.subplot() # Defines ax variable by creating an empty plot
    
    # Set the tick labels font
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(font_size)
    
    plt.xlabel('Sense', **axis_font)
    plt.ylabel('PredictionAccuracy', **axis_font)
    
    plt.title(title, **title_font)
    
    #Plot the Image!
    # plt.bar(range(0,n), np.sort(mean_r[0:n]), color = 'darkred')
    plt.bar(range(0,n), np.sort(mean_r_nonan)[::-1], color = 'darkred')  
    
    #ylim
    plt.ylim(-0.2, 0.6) 
    
    #save figure
    if figure_path is not None:
        if not pretrained:
            plt.savefig( figure_path + filename + '_untrained.png')
        else:
            plt.savefig(figure_path + filename + '.png')

    sns.reset_defaults()
    plt.close()


def p2r(p, n):
    t = stats.t.ppf(1-p, n-2);
    r = (t**2/((t**2)+(n-2))) ** 0.5;
    return r