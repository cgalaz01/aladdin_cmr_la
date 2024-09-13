import os
from typing import List, Optional, Tuple

import numpy as np
from scipy import spatial, ndimage
from skimage.segmentation import find_boundaries
from skimage import morphology, measure
from skimage.filters import gaussian
import pyvista as pv

import SimpleITK as sitk

from atlas_generation import (affine_register_cases, inverse_displacement_field,
                              select_la, smooth_mesh, nonrigid_register_cases)


def load_atlas(atlas_path: str) -> Tuple[sitk.Image, sitk.Image, sitk.Image, sitk.Image, pv.PolyData]:
    """
    Load the reference image and segmentation and, atlas image, segmentation
    and mesh.

    Parameters
    ----------
    atlas_path : str
        Path to the atlas files.

    Returns
    -------
    ref_image : sitk.Image
        Reference image for the atlas.
    ref_seg : sitk.Image
        Reference segmentation for the atlas.
    atlas_seg : sitk.Image
        Atlas segmentation.
    atlas_mesh : pv.PolyData
        Atlas mesh.

    """
    ref_image = sitk.ReadImage(os.path.join(base_path, 'reference_image.nii.gz'))
    ref_seg = sitk.ReadImage(os.path.join(base_path, 'reference_segmentation.nii.gz'))
    atlas_seg = sitk.ReadImage(os.path.join(base_path, 'atlas_segmentation.nii.gz'))
    
    atlas_mesh = pv.read(os.path.join(base_path, 'atlas_mesh.vtk'))

    return ref_image, ref_seg, atlas_seg, atlas_mesh


def load_patient(data_path: str, atlas_segmentation: sitk.Image) -> \
                Tuple[sitk.Image, sitk.Image, List[sitk.Image]]:
    """
    Load patient images, segmentations and displacements.
    Note displacement values are expected to be in physical space rather than
    image space.

    Parameters
    ----------
    data_path : str
        Path to the patient data.
    atlas_segmentation : sitk.Image
        Atlas segmentation used to copy the metadata.

    Returns
    -------
    image : sitk.Image
        Loaded patient image.
    segmentation : sitk.Image
        Loaded patient segmentation.
    dvfs : List[sitk.Image]
        List of loaded displacement vector fields across the cardiac cycle.

    """
    image_path = os.path.join(data_path, 'images', '00.nii.gz')
    image = sitk.ReadImage(image_path)
    segmentation_path = os.path.join(data_path, 'segmentations', '00.nii.gz')
    segmentation = sitk.ReadImage(segmentation_path)
    
    dvf_files = sorted(os.listdir(os.path.join(data_path, 'displacements', 'full')))     
    dvfs = []
    for i in range(len(dvf_files)):
        dvf = sitk.ReadImage(os.path.join(data_path, 'displacements', 'full',
                                          dvf_files[i]))
        dvfs.append(dvf)
        
    # Lazy physical alignment
    image.CopyInformation(atlas_segmentation)
    segmentation.CopyInformation(atlas_segmentation)
    for k in range(len(dvfs)):
        dvfs[k].CopyInformation(atlas_segmentation)
        
    return image, segmentation, dvfs


def transform_images_and_dvfs(image: sitk.Image, segmentation: sitk.Image,
                              dvfs: List[sitk.Image], transformation: sitk.Transform) -> \
                Tuple[sitk.Image, sitk.Image, List[sitk.Image]]:
    """
    Transform patient images and displacement vector fields using a given
    transformation.

    Parameters
    ----------
    image : sitk.Image
        Patient image to be transformed.
    segmentation : sitk.Image
        Patient segmentation to be transformed.
    dvfs : List[sitk.Image]
        List of displacement vector fields to be transformed.
    transformation : sitk.Transform
        Transformation to be applied.

    Returns
    -------
    image : sitk.Image
        Transformed patient image.
    segmentation : sitk.Image
        Transformed patient segmentation.
    dvfs : List[sitk.Image]
        List of transformed displacement vector fields.

    """
    image = sitk.Resample(image, transformation, sitk.sitkLinear,
                          0.0, image.GetPixelID())
    segmentation = sitk.Resample(segmentation, transformation, sitk.sitkNearestNeighbor,
                                 0, segmentation.GetPixelID())
    
    for i in range(len(dvfs)):
        # Note: use inverse transformation as it's transformation from fixed to moving
        # but we want moving to fixed for transforming the vector
        dvfs[i] = transform_vector(vector_sitk=dvfs[i], transformation=transformation.GetInverse())        
        dvfs[i] = transform_vector_as_image(vector_sitk=dvfs[i], transformation=transformation)
        
    return image, segmentation, dvfs


def get_segmentation_edge(segmentation: sitk.Image) -> sitk.Image:
    """
    Extract the edge of a binary segmentation image.

    Parameters
    ----------
    segmentation : sitk.Image
        The input segmentation image.

    Returns
    -------
    edge_seg : sitk.Image
        The edge of the segmentation image.

    """
    edge_seg = sitk.GetArrayFromImage(segmentation)
    edge_seg = np.swapaxes(edge_seg, 0, 2)
    
    edge_seg = find_boundaries(edge_seg, connectivity=3,
                               mode='inner', background=0).astype(edge_seg.dtype)
    
    # xy dilation
    disk = morphology.disk(radius=2)
    disk = np.expand_dims(disk, axis=-1)
    edge_seg_1 = ndimage.binary_dilation(edge_seg, disk, iterations=1).astype(int)
    
    dilation_radius = 1
    sphere = morphology.ball(radius=dilation_radius)
    edge_seg_2 = ndimage.binary_dilation(edge_seg, sphere, iterations=1).astype(int)
    
    edge_seg = np.maximum(edge_seg_1, edge_seg_2)
    
    edge_seg = np.swapaxes(edge_seg, 0, 2)
    edge_seg = sitk.GetImageFromArray(edge_seg)
    edge_seg.CopyInformation(segmentation)
    
    return edge_seg

    
def compute_meshes(segmentation: sitk.Image, displacement_fields: List[sitk.Image],
                   image: Optional[sitk.Image] = None, smooth: bool = False) -> List[pv.PolyData]:
    """
    Compute the mesh from the segmentation and map to it the displacement vector
    fields.

    Parameters
    ----------
    segmentation : sitk.Image
        The segmentation image.
    displacement_fields : List[sitk.Image]
        The list of displacement vector fields.
    image : Optional[sitk.Image], optional
        The optional image. Default is None.
    smooth : bool, optional
        Flag indicating whether to smooth the meshes. Default is False.

    Returns
    -------
    meshes : List[pv.PolyData]
        The list of computed meshes.

    """
    segmentation = sitk.GetArrayFromImage(segmentation)
    segmentation = np.swapaxes(segmentation, 0, 2)
    # Make sure LA is only present
    segmentation[segmentation != 1] = 0
    
 
    verts, faces, normals, values = measure.marching_cubes(segmentation, level=0.5,
                                                           gradient_direction='descent',
                                                           allow_degenerate=False,
                                                           method='lewiner')
    vfaces = np.column_stack((np.ones(len(faces),) * 3, faces)).astype(int)
    
    meshes = []
    for displacement_field in displacement_fields:
        displacement_field = sitk.GetArrayFromImage(displacement_field)
        displacement_field = np.swapaxes(displacement_field, 0, 2)
        # Convert the displacement field from physical space to image space
        spacing = segmentation.GetSpacing()
        for axis in range(len(spacing)):
            displacement_field[..., axis] /= spacing[axis]
        
        if smooth:
            displacement_field = gaussian(displacement_field, sigma=2, preserve_range=True,
                                          truncate=4.0, channel_axis=-1)
              
        mesh = pv.PolyData(verts, vfaces)
        mesh['Normals'] = normals
        mesh['values'] = values
        mesh = mesh.clean(tolerance=1e-7).triangulate()
            
        if smooth:
            mesh = smooth_mesh(mesh)
        
        x = np.linspace(0, displacement_field[..., 0].shape[0] -1, displacement_field[..., 0].shape[0])
        y = np.linspace(0, displacement_field[..., 1].shape[1] -1, displacement_field[..., 1].shape[1])
        z = np.linspace(0, displacement_field[..., 2].shape[2] -1, displacement_field[..., 2].shape[2])
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=-1)
        dvf_values = np.stack((displacement_field[..., 0].flatten(),
                               displacement_field[..., 1].flatten(),
                               displacement_field[..., 2].flatten()), axis=-1)
        
        dvf_mesh = pv.PolyData(points)
        dvf_mesh.point_data['dvf'] = dvf_values
        
        if not image is None:
            image_values = sitk.GetArrayFromImage(image)
            image_values = np.swapaxes(image_values, 0, -1)
            image_values = image_values.flatten()
            dvf_mesh.point_data['img'] = image_values
            
        mesh = mesh.interpolate(dvf_mesh, radius=2.0, sharpness=1.0, pass_cell_data=False)
            
        meshes.append(mesh)
    
    return meshes


def metric_tensor(nodes: np.ndarray) -> np.ndarray:
    """
    Calculate the metric tensor for a given set of nodes.

    Parameters
    ----------
    nodes : np.ndarray
        Array of shape (3, :) representing the coordinates of the nodes.

    Returns
    -------
    g : np.ndarray
        Array of shape (3, 3) representing the metric tensor.

    """
    e = 1e-9
    g_1 = nodes[1, :] - nodes[0, :]
    g_2 = nodes[2, :] - nodes[0, :]
    
    g_3 = np.cross(g_1, g_2)
    g_3n = np.linalg.norm(g_3)
    if g_3n == 0:
        g_3n = e
    g_3 /= g_3n
    
    v = np.row_stack((g_1, g_2, g_3))
    g = np.dot(v, v.T)
            
    return g


def local_cartesian(nodes: np.ndarray) -> np.ndarray:
    """
    Calculate the local Cartesian coordinate system for a given set of nodes.

    Parameters
    ----------
    nodes : np.ndarray
        Array of shape (3, :) representing the coordinates of the nodes.

    Returns
    -------
    es : np.ndarray
        Array of shape (3, 3) representing the local Cartesian coordinate system.

    """
    e1 = nodes[1, :] - nodes[0, :]
    e1 /= np.linalg.norm(e1)
    
    v = nodes[2, :] - nodes[0, :]
    e2 = v - np.dot(v, e1) * e1
    e2 /= np.linalg.norm(e2)
    
    e3 = np.cross(e1, e2)
    e3 /= np.linalg.norm(e3)
    
    es = np.row_stack((e1, e2, e3))
    
    return es


def calc_area_ratio(nodes: np.ndarray, deformed: np.ndarray) -> np.ndarray:
    """
    Calculate the area ratio between two triangles defined by the input nodes
    and deformed nodes.

    Parameters
    ----------
    nodes : np.ndarray
        Array of shape (3, :) representing the coordinates of the nodes of the
        original triangle.
    deformed : np.ndarray
        Array of shape (3, :) representing the coordinates of the nodes of the
        deformed triangle.

    Returns
    -------
    area_ratio : np.ndarray
        Array representing the area ratio between the two triangles.

    """
    E1 = nodes[1, :] - nodes[0, :]
    v = nodes[2, :] - nodes[0, :]
    E2 = v - np.dot(v, E1) * E1
    e1 = deformed[1, :] - deformed[0, :]
    v = deformed[2, :] - deformed[0, :]
    e2 = v - np.dot(v, e1) * e1
    area_ratio = (np.linalg.norm(e1) * np.linalg.norm(e2)) / (np.linalg.norm(E1) * np.linalg.norm(E2))
    
    return area_ratio


def calculate_principal_strains_covariant(coords: np.ndarray, deformed: np.ndarray,
                                          els: np.ndarray, norms: np.ndarray) -> \
                Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the principal strains and directions in covariant form for a given
    set of coordinates, deformed coordinates, element indices, and surface normals.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, 3) representing the original coordinates.
    deformed : np.ndarray
        Array of shape (N, 3) representing the deformed coordinates.
    els : np.ndarray
        Array of shape (M,) representing the indices of the mesh faces.
    norms : np.ndarray
        Array of shape (M, 3) representing the surface normals of the elements.

    Returns
    -------
    strains : np.ndarray
        Array of shape (M, 3, 3) representing the strains for each element.
    Epr : np.ndarray
        Array of shape (M, 3) representing the principal strain values for each
        element.
    vpr : np.ndarray
        Array of shape (M, 3, 3) representing the principal strain directions
        for each element.
    area_ratio : np.ndarray
        Array of shape (M,) representing the area ratio for each element.

    """
    strains = []
    Epr = []
    vpr = []
    area_ratio = np.zeros(len(els))
    
    for i in range(len(els)):
        # Calculate metric tensors and Lagrange strain in curvilinear coords
        Gij = metric_tensor(coords[els[i], :])
        gij = metric_tensor(deformed[els[i], :])
        E = 0.5 * (gij - Gij)
        
        # Map strain back to local Cartesian
        es = local_cartesian(coords[els[i], :])
        E = np.dot(np.dot(es.T, E), es)
        
        # Calculate principal strains and directions
        d, v = np.linalg.eig(E)

        # Order by highest absolute value
        sort_indexes = np.argsort(np.abs(d))[::-1]
        
        d = np.asarray([d[sort_indexes[0]], d[sort_indexes[1]], d[sort_indexes[2]]])
        v = np.asarray([v[:, sort_indexes[0]], v[:, sort_indexes[1]], v[:, sort_indexes[2]]])
        
        if (np.dot(norms[i], v[2])/(np.linalg.norm(norms[i]) * np.linalg.norm(v[2])) <
            np.dot(norms[i], -v[2])/(np.linalg.norm(norms[i]) * np.linalg.norm(-v[2]))):
            v = -v
        
        strains.append(E)
        Epr.append(d)
        vpr.append(v)
        
        # area ratio
        area_ratio[i] = calc_area_ratio(coords[els[i], :], deformed[els[i], :])
    
    strains = np.asarray(strains)
    Epr = np.asarray(Epr)
    vpr = np.asarray(vpr)
    area_ratio = np.asarray(area_ratio)
    
    return strains, Epr, vpr, area_ratio


def transform_vector(vector_sitk: sitk.Image, transformation: sitk.Transform) -> sitk.Image:
    """
    Transform a vector using a given transformation.

    Parameters
    ----------
    vector_sitk : sitk.Image
        The input vector image to be transformed.
    transformation : sitk.Transform
        The transformation to apply to the vector image.

    Returns
    -------
    vector_transformed : sitk.Image
        The transformed vector image.

    """
    vector = sitk.GetArrayFromImage(vector_sitk)
    vector = np.swapaxes(vector, axis1=0, axis2=2)
    vector_transformed = np.zeros_like(vector)
    for i in range(vector.shape[0]):
        for j in range(vector.shape[1]):
            for k in range(vector.shape[2]):
                physical_point = vector_sitk.TransformIndexToPhysicalPoint((i, j, k))
                vector_transformed[i, j, k] = transformation.TransformVector(vector[i, j, k],
                                                                             physical_point)
                
    vector_transformed = np.swapaxes(vector_transformed, axis1=0, axis2=2)
    vector_transformed = sitk.GetImageFromArray(vector_transformed)
    vector_transformed.CopyInformation(vector_sitk)      
    
    return vector_transformed


def transform_vector_as_image(vector_sitk: sitk.Image, transformation: sitk.Transform) -> sitk.Image:
    """
    Transform a vector as if it was an image using a given transformation.

    Parameters
    ----------
    vector_sitk : sitk.Image
        The input vector image to be transformed.
    transformation : sitk.Transform
        The transformation to apply to the vector image.

    Returns
    -------
    transformed_vector : sitk.Image
        The transformed vector image.

    """
    interpolator = sitk.sitkLinear
    default_value = 0.0
    transformed_vector = sitk.Resample(vector_sitk, vector_sitk, transformation,
                                       interpolator, default_value)
    
    return transformed_vector


def scale_points(points: np.ndarray, scale: List[float]) -> np.ndarray:
    """
    Scale the given points by the specified scale factors.

    Parameters
    ----------
    points : np.ndarray
        The input array of points to be scaled.
    scale : List[float]
        The list of scale factors to apply to each dimension of the points.

    Returns
    -------
    scaled_points : np.ndarray
        The array of scaled points.

    """
    scaled_points = []
    for i in range(len(points)):
        new_points = []
        for d in range(len(points[i])):
            new_points.append(points[i][d] * scale[d])
        scaled_points.append(new_points)
    
    scaled_points = np.asarray(scaled_points)
    return scaled_points
        
    
def deform_points(points: np.ndarray, dvf: np.ndarray, scale: List[float]) -> np.ndarray:
    """
    Deform the given points using the specified displacement vector field and
    scale factors (physical space).

    Parameters
    ----------
    points : np.ndarray
        The input array of points to be deformed.
    dvf : np.ndarray
        The displacement vector field (DVF) representing the deformation to be
        applied to the points.
    scale : List[float]
        The list of scale factors to apply to each dimension of the points.

    Returns
    -------
    deformed_points : np.ndarray
        The array of deformed points.

    """
    deformed_points = []
    for i in range(len(points)):
        new_points = []
        for d in range(len(points[i])):
            new_points.append(points[i][d] + (dvf[i][d] * scale[d]))    
        deformed_points.append(new_points)
        
    deformed_points = np.asarray(deformed_points)
    
    return deformed_points
    

def compute_strains(meshes: List[pv.PolyData], segmentation: sitk.Image) -> List[pv.PolyData]:
    """
    Compute principal surface strains on the meshes across the cardiac cycle.

    Parameters
    ----------
    meshes : List[pv.PolyData]
        The list of meshes to compute strains for.
    segmentation : sitk.Image
        The segmentation image used to convert points to physical space.

    Returns
    -------
    updated_meshes : List[pv.PolyData]
        The list of meshes with updated strain information.

    """
    updated_meshes = []
    for mesh in meshes:
            
        disp = mesh.point_data['dvf']
        umag = np.sqrt(disp[:, 0]**2 + disp[:, 1]**2 + disp[:, 2]**2) # magnitude of displacements
        mesh.point_data['umag'] = umag
        
        # Convert points from image to physical space
        scaled_points = scale_points(mesh.points, scale=segmentation.GetSpacing())
        deformed_points = deform_points(scaled_points, mesh.point_data['dvf'],
                                        scale=segmentation.GetSpacing())

        
        mesh = mesh.compute_normals(cell_normals=True, point_normals=False, flip_normals=True)
        cell_norms = mesh.cell_data['Normals']
        # mesh.faces -> first value is the number of points of the face (so reshape to 4, for triangle face)
        strain, Epr, vpr, area_ratio = calculate_principal_strains_covariant(
            coords=scaled_points, deformed=deformed_points, els=mesh.faces.reshape(-1, 4)[:, 1:], norms=cell_norms)
        
        # Interpolate the strains to the vertices of the mesh
        face_centers = mesh.cell_centers()
        #face_centers.point_data['strain'] = strain
        face_centers.point_data['Epr'] = Epr
        face_centers.point_data['vpr_0'] = vpr[:, 0]
        face_centers.point_data['vpr_1'] = vpr[:, 1]
        face_centers.point_data['vpr_2'] = vpr[:, 2]
        strain_mesh = mesh.interpolate(face_centers, radius=2.0, sharpness=1.0)
        #point_strain = strain_mesh.point_data['strain'].reshape((-1, 3, 3))
        
        # save everything as a mesh
        #mesh.point_data['strain'] = point_strain
        mesh.point_data['Epr'] = strain_mesh.point_data['Epr']
        mesh.point_data['vpr_0'] = strain_mesh.point_data['vpr_0']
        mesh.point_data['vpr_1'] = strain_mesh.point_data['vpr_1']
        mesh.point_data['vpr_2'] = strain_mesh.point_data['vpr_2']
        
        mesh.cell_data['Epr_cell'] = Epr
        mesh.cell_data['vpr_cell_0'] = vpr[:, 0]
        mesh.cell_data['vpr_cell_1'] = vpr[:, 1]
        mesh.cell_data['vpr_cell_2'] = vpr[:, 2]
        #mesh.cell_data['area_ratio'] = area_ratio
        mesh.cell_data['cell_normals'] = cell_norms 
        updated_meshes.append(mesh)
        
    return updated_meshes


def warp_mesh(mesh: pv.PolyData, displacement_field_image: sitk.DisplacementFieldTransform) -> \
                pv.PolyData:
    """
    Warp the given mesh using the provided displacement vector field.

    Parameters
    ----------
    mesh : pv.PolyData
        The input mesh to be warped.
    displacement_field_image : sitk.DisplacementFieldTransform
        The displacement field used for warping the mesh.

    Returns
    -------
    mesh : pv.PolyData
        The warped mesh.

    """
    dvf = inverse_displacement_field(displacement_field_image)
    dvf = sitk.GetArrayFromImage(dvf.GetDisplacementField())
    dvf = np.swapaxes(dvf, 0, 2)
    
    vector_displacement = []
    for point in mesh.points:
        point = [int(i) for i in point]
        point_dvf = dvf[point[0], point[1], point[2], :]
        vector_displacement.append(point_dvf)    

    vector_displacement = np.asarray(vector_displacement)
    mesh.point_data['registration_dvf'] = vector_displacement
    mesh = mesh.warp_by_vector(vectors='registration_dvf', factor=1.0, inplace=False)
    del mesh.point_data['registration_dvf']
    
    return mesh
    

def interpolate_mesh(meshes: List[pv.PolyData], atlas_mesh: pv.PolyData) -> List[pv.PolyData]:
    """
    Interpolate the given meshes to match the shape of the atlas mesh.
    It is assumed that the meshes are already (closely) aligned to the atlas mesh.

    Parameters
    ----------
    meshes : List[pv.PolyData]
        The list of meshes to be interpolated.
    atlas_mesh : pv.PolyData
        The atlas mesh used as a reference for interpolation.

    Returns
    -------
    meshes : List[pv.PolyData]
        The list of interpolated meshes.

    """
    # Get the corresponding registered mesh with the atlas and use indexes
    # to make sure values are mapped correctly to the points
    kd_tree = spatial.KDTree(meshes[0].points)
    distances, indexes = kd_tree.query(atlas_mesh.points, k=1)
     
    for i in range(len(meshes)):
        meshes[i] = atlas_mesh.interpolate(meshes[i], sharpness=1, radius=2.0,
                                           strategy='null_value',
                                           null_value=0.0,
                                           n_points=None)
    
    return meshes


if __name__ == '__main__':
    output_folder = '_registration_output'
    atlas_path = os.path.join('_atlas_output')
    base_path = os.path.join('..', '..', 'data_nn', 'train')   
    patients = sorted(os.listdir(base_path))
    
    print('Loading atlas data...')
    ref_image, ref_seg, atlas_seg, atlas_mesh = load_atlas(atlas_path)
    atlas_edge_seg = get_segmentation_edge(atlas_seg)
    
    for patient in patients:
        print('Registering case: ' + patient)
        
        print('Loading patient data...')
        image, segmentation, dvfs = load_patient(os.path.join(base_path, patient), atlas_seg)
                
        print('Executing affine registration...')
        affine_transformation = affine_register_cases([image], [segmentation],
                                                      ref_image, ref_seg)[0]
        # Apply the transformation
        image, segmentation, dvfs = transform_images_and_dvfs(image, segmentation, dvfs, affine_transformation)
        
        
        print('Executing mesh computation...')
        meshes = compute_meshes(segmentation, dvfs, image=image, smooth=True)
        print('Executing strain computation...')
        meshes = compute_strains(meshes, segmentation)
        
        # We only need LA now
        segmentation = select_la(segmentation, segmentation=None)
        image = select_la(image, segmentation=segmentation)
        # Get the contour from the segmentations to simplify the registration
        segmentation = get_segmentation_edge(segmentation)
        
        print('Executing non-rigid registration...')
        displacement_field = nonrigid_register_cases(image, segmentation,
                                                     atlas_seg, atlas_edge_seg,
                                                     initial_alignment=None)
                                                     
        # Apply the displacement to all the meshes across the cardiac cycle
        for i in range(len(meshes)):
            meshes[i] = warp_mesh(meshes[i], displacement_field)
            
        
        # Align meshes to the atlas mesh
        print('Executing mesh interpolation...')
        meshes = interpolate_mesh(meshes, atlas_mesh)
            
        print('Saving meshes')
        prefix_mesh_folder = os.path.join(output_folder, patient)
        os.makedirs(prefix_mesh_folder, exist_ok=True)
        for i in range(len(meshes)):
            meshes[i].save(os.path.join(prefix_mesh_folder, f'mesh{i:02d}.vtk'))
            