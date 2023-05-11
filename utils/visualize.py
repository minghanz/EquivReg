import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision.utils import save_image
import common
try:
    import open3d as o3d
except:
    print("Warning: failed to import open3d, some function may not be used. ")


def visualize_data(data, data_type, out_file, data2=None, info=None, c1=None, c2=None, show=False, s1=5, s2=5):
    r''' Visualizes the data with regard to its type.

    Args:
        data (tensor): batch of data
        data_type (string): data type (img, voxels or pointcloud)
        out_file (string): output file
    '''
    if data_type == 'img':
        if data.dim() == 3:
            data = data.unsqueeze(0)
        save_image(data, out_file, nrow=4)
    elif data_type == 'voxels':
        visualize_voxels(data, out_file=out_file)
    elif data_type == 'pointcloud':
        visualize_pointcloud(data, out_file=out_file, points2=data2, info=info, c1=c1, c2=c2, show=show, s1=s1, s2=s2)
        # display_open3d(data)
    elif data_type is None or data_type == 'idx':
        pass
    else:
        raise ValueError('Invalid data_type "%s"' % data_type)


def visualize_voxels(voxels, out_file=None, show=False):
    r''' Visualizes voxel data.

    Args:
        voxels (tensor): voxel data
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    voxels = np.asarray(voxels)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    voxels = voxels.transpose(2, 0, 1)
    ax.voxels(voxels, edgecolor='k')
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)


def display_open3d(template, source=None, transformed_source=None):
    to_vis = []
    template_ = o3d.geometry.PointCloud()
    template_.points = o3d.utility.Vector3dVector(template)
    template_.paint_uniform_color([1, 0, 0])
    to_vis.append(template_)
    if source is not None:
        source_ = o3d.geometry.PointCloud()
        source_.points = o3d.utility.Vector3dVector(source + np.array([0,0,0]))
        source_.paint_uniform_color([0, 1, 0])
        to_vis.append(source_)
    if transformed_source is not None:
        transformed_source_ = o3d.geometry.PointCloud()
        transformed_source_.points = o3d.utility.Vector3dVector(transformed_source)
        transformed_source_.paint_uniform_color([0, 0, 1])
        to_vis.append(transformed_source_)
    o3d.visualization.draw_geometries(to_vis)
	# o3d.visualization.draw_geometries([template_, source_, transformed_source_])

def hat(v):
    mat = np.array([[0, -v[2], v[1]], 
                    [v[2], 0, -v[0]], 
                    [-v[1], v[0], 0]])
    return mat
def visualize_feat_as_vec_field(points, feature, idx=list(range(10)), rotmat=None, 
                                out_file=None, show=False, size=7):
    for ii in idx:
        out_1_feat_samp = feature[ii] # 3-vector
        mat_1 = hat(out_1_feat_samp)
        field_1 = points.dot(mat_1.T)
        visualize_pointcloud(points, field_1, out_file, show, s1=size)

        if rotmat is not None:
            field_1 = field_1.dot(rotmat)
            points = points.dot(rotmat)
            visualize_pointcloud(points, field_1, out_file, show, s1=size)
    return

def visualize_pointcloud(points, normals=None,
                         out_file=None, show=False, 
                         points2=None, info=None, 
                         c1=None, c2=None, cm1='viridis', cm2='viridis', s1=5, s2=5):
    r''' Visualizes point cloud data.

    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    points = np.asarray(points)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    if c1 is not None:
        cmap1 = plt.get_cmap(cm1)   # viridis, magma
        ax.scatter(points[:, 2], points[:, 0], points[:, 1], s=s1, c=c1, cmap=cmap1)
    else:
        ax.scatter(points[:, 2], points[:, 0], points[:, 1])
    if points2 is not None:
        if c2 is not None:
            cmap2 = plt.get_cmap(cm2)
            ax.scatter(points2[:, 2], points2[:, 0], points2[:, 1], s=s2, c=c2, cmap=cmap2, marker='^')
        else:
            ax.scatter(points2[:, 2], points2[:, 0], points2[:, 1], 'r')
    if normals is not None:
        ax.quiver(
            points[:, 2], points[:, 0], points[:, 1],
            normals[:, 2], normals[:, 0], normals[:, 1],
            length=0.8, color='gray', linewidth=0.8
        )
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=30, azim=45)
    if info is not None:
        plt.title("{}".format(info))

    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)


def visualise_projection(
        self, points, world_mat, camera_mat, img, output_file='out.png'):
    r''' Visualizes the transformation and projection to image plane.

        The first points of the batch are transformed and projected to the
        respective image. After performing the relevant transformations, the
        visualization is saved in the provided output_file path.

    Arguments:
        points (tensor): batch of point cloud points
        world_mat (tensor): batch of matrices to rotate pc to camera-based
                coordinates
        camera_mat (tensor): batch of camera matrices to project to 2D image
                plane
        img (tensor): tensor of batch GT image files
        output_file (string): where the output should be saved
    '''
    points_transformed = common.transform_points(points, world_mat)
    points_img = common.project_to_camera(points_transformed, camera_mat)
    pimg2 = points_img[0].detach().cpu().numpy()
    image = img[0].cpu().numpy()
    plt.imshow(image.transpose(1, 2, 0))
    plt.plot(
        (pimg2[:, 0] + 1)*image.shape[1]/2,
        (pimg2[:, 1] + 1) * image.shape[2]/2, 'x')
    plt.savefig(output_file)
