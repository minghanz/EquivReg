import os
import numpy as np
import utils.binvox_rw as binvox_rw
import logging

class Field(object):
    ''' Data fields class.
    '''

    def load(self, data_path):
        ''' Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        '''
        raise NotImplementedError

    def check_complete(self, files):
        ''' Checks if set is complete.

        Args:
            files: files
        '''
        raise NotImplementedError
    
class PointsField(Field):
    ''' Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided

    '''
    def __init__(self, file_name, transform=None, unpackbits=False):
        self.file_name = file_name
        self.transform = transform
        self.unpackbits = unpackbits

    def load(self, model_path):
        
        # Load data
        file_path = os.path.join(model_path, self.file_name)
        points_dict = np.load(file_path)

        # Points
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
        else:
            points = points.astype(np.float32)

        # Occupancies
        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        # Output dict
        data = {
            None: points,
            'occ': occupancies,
        }

        if self.transform is not None:
            data = self.transform(data)

        return data
    
class VoxelsField(Field):
    ''' Voxel field class.

    It provides the class used for voxel-based data.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path):
        
        file_path = os.path.join(model_path, self.file_name)

        with open(file_path, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f)
        voxels = voxels.data.astype(np.float32)

        if self.transform is not None:
            voxels = self.transform(voxels)

        return voxels

    def check_complete(self, files):

        complete = (self.file_name in files)
        return complete
    
class PointCloudField(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        with_transforms (bool): whether scaling and rotation dat should be
            provided
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load_dict(self, pointcloud_dict):

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        # print("points.shape", points.shape)     # 100000, 3
        data = {
            None: points,
            'normals': normals,
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def load_array(self, pointcloud_array):
        data = {None: pointcloud_array}

        if self.transform is not None:
            data = self.transform(data)

        return data

    def load(self, model_path):
        
        file_path = os.path.join(model_path, self.file_name)

        pointcloud_dict = np.load(file_path)

        if isinstance(pointcloud_dict, np.lib.npyio.NpzFile):
            data = self.load_dict(pointcloud_dict)
        elif isinstance(pointcloud_dict, np.ndarray):
            data = self.load_array(pointcloud_dict)
        else:
            raise ValueError('pointcloud file content {} unexpected: {}'.format(type(pointcloud_dict), file_path))

        return data

    def check_complete(self, files):
        
        complete = (self.file_name in files)
        return complete
    
class RotationField(Field):
    '''It provides the field used for a rotation transformation. 
    When benchmarking registration performance, 
    it is useful to have fixed initial transformations instead of random ones.
    '''
    def __init__(self, file_name) -> None:
        super().__init__()
        self.file_name = file_name

    def load(self, model_path):
        
        file_path = os.path.join(model_path, self.file_name)
        data = np.load(file_path)
        assert 'T' in data and 'deg' in data, data
        data_out = {
            None: data['T'], 
            'deg': data['deg'],
        }

        return data_out
    
class TransformationField(Field):
    '''It provides the field used for a rigid body transformation. 
    When benchmarking registration performance, 
    it is useful to have fixed initial transformations instead of random ones.
    '''
    def __init__(self, file_name) -> None:
        super().__init__()
        self.file_name = file_name

    def load(self, model_path):
        
        file_path = os.path.join(model_path, self.file_name)
        T = np.load(file_path)
        return T
    
class IndexField(Field):
    ''' Basic index field.'''

    def check_complete(self, files):
        
        return True
    
class CategoryField(Field):
    ''' Basic category field.'''

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True