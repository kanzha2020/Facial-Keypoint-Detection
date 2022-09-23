import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        #key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        key_pts = self.key_pts_frame.iloc[idx, 1:].to_numpy() # Modified by user
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
            
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0


        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        left,top=key_pts.min(axis=0)#Added by users
        right,bottom=key_pts.max(axis=0)#Added by users
        top = top.astype(int)#Added by users
        left = left.astype(int)#Added by users
        right = right.astype(int)#Added by users
        bottom = bottom.astype(int)#Added by users
        face_h = bottom-top+2
        face_w = right-left+2
        #print('h',h)
        #print('w',w)
        #print('face_h',face_h)
        #print('face_w',face_w)
        ### Added by user ########################
        #if face_h > face_w:
        #    new_h = h* (224/face_h)
        #    new_w = w* (224/face_h)
        #else:
        #    new_h = h* (224/face_w)
        #    new_w = w* (224/face_w)
        ##########################################
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        #print('new_h',new_h)
        #print('new_w',new_w)
    
        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        left_real,top_real=key_pts.min(axis=0)#Added by users
        
        #top = np.random.randint(max(0,top_real-50), min(top_real+100,h - new_h))
        #left = np.random.randint(max(0,left_real-50), min(left_real+100,w - new_w))
        
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        #left,top=key_pts.min(axis=0)#Added by users
        #top = top.astype(int)-10#Added by users
        #left = left.astype(int)-10#Added by users
        #print('left',left)
        #print('top',top)
        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}
    
# Added by user
class FaceCrop(object):
    """Crop the face image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        left,top=key_pts.min(axis=0)#Added by users
        right,bottom=key_pts.max(axis=0)#Added by users
        top = top.astype(int)#Added by users
        left = left.astype(int)#Added by users
        right = right.astype(int)#Added by users
        bottom = bottom.astype(int)#Added by users
        face_h = bottom-top+0
        face_w = right-left+0
        final_hw = max(face_h,face_w)
        final_hw = final_hw+final_hw/2
        final_hw = final_hw.astype(int)
        
        top = top-final_hw/8#Added by users
        top = top.astype(int)#Added by users
        left = left-final_hw/8#Added by users
        left = left.astype(int)#Added by users
        
        top = max(0,top)#Added by users
        left = max(0,left)#Added by users
        #print('left',left)
        #print('top',top)
        image = image[top: top + final_hw,
                      left: left + final_hw]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}
    

# Added by user
class FaceRotate(object):
    """Crop the face image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        
        angle = np.random.randint(0, 30)
        print('rotation angle',angle)
        
        M = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
        image_rotate = cv2.warpAffine(image,M,(w,h))

        return {'image': image_rotate, 'keypoints': key_pts}