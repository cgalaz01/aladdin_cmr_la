from typing import Optional, List, Tuple, Union

import numpy as np
from scipy import ndimage

import SimpleITK as sitk


class DataAugmentation():
    
    def __init__(self, seed: Union[int, None]):
        """
        Initializes the DataAugmentation class.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator. If None, the random number
            generator will not be seeded.

        """
        self.random_generator = np.random.RandomState(seed)
        
        self.min_z_rotation_degrees = -30
        self.max_z_rotation_degrees = 30
    
        self.min_gaussian_blur_sigma = 0.0
        self.max_gaussian_blur_sigma = 1.0
        
        self.gaussian_scale = 0.01
        
        self.rician_mean = 0.0
        self.rician_std = 0.05
        
        self.max_abs_x_scale = 0.15
        self.max_abs_y_scale = 0.15
        self.max_abs_z_scale = 0.15
        
        self.max_intensity_scale = 0.01
        
        self.coeff_range = [0, 0.1]
        self.degree = 3
        
    
    @staticmethod
    def _matrix_from_axis_angle(a: Tuple[float]) -> np.ndarray:
        """ 
        Compute the rotation matrix from axis-angle. (Rodrigues' formula)
        
        
        Parameters
        ----------
        a : Tuple[float]
            Axis of rotation and rotation angle: (x, y, z, angle).

        Returns
        -------
        R : np.ndarray
            Rotation matrix.
        """
        ux, uy, uz, theta = a
        c = np.cos(theta)
        s = np.sin(theta)
        ci = 1.0 - c
        R = np.array([[ci * ux * ux + c,
                       ci * ux * uy - uz * s,
                       ci * ux * uz + uy * s],
                      [ci * uy * ux + uz * s,
                       ci * uy * uy + c,
                       ci * uy * uz - ux * s],
                      [ci * uz * ux - uy * s,
                       ci * uz * uy + ux * s,
                       ci * uz * uz + c],
                      ])
    
        return R


    @staticmethod
    def _rotate_z_axis(image: sitk.Image, degrees: float, is_labels: bool) -> \
                Tuple[sitk.Image, sitk.Euler3DTransform]:
        """
        Rotates the given image around the image z-axis by the given degrees.

        Parameters
        ----------
        image : sitk.Image
            The image to rotate around the image space z-axis.
        degrees : float
            The rotation amount in degrees.
        is_labels : bool
            Whether the image is a label (nearest neighbor interpolation) or
            not (linear interpolation).

        Returns
        -------
        rotated_image : sitk.image
            The rotated image.
        transformation : sitk.Euler3DTransform
            The transformation matrix used to rotate the image.

        """
        # Adapted from:
        #   https://stackoverflow.com/questions/56171643/simpleitk-rotation-of-volumetric-data-e-g-mri
        
        radians = np.deg2rad(degrees)
        
        # Find image centre
        width, height, depth = image.GetSize()
        physical_centre = image.TransformIndexToPhysicalPoint((width // 2,
                                                               height // 2,
                                                               depth // 2))
        
        direction = image.GetDirection()
        axis_angle = (direction[2], direction[5], direction[8], radians)
        rotation_matrix = DataAugmentation._matrix_from_axis_angle(axis_angle)
        
        # Construct transfomration matrix
        transformation = sitk.Euler3DTransform()
        transformation.SetCenter(physical_centre)
        transformation.SetMatrix(rotation_matrix.flatten().tolist())
        
        
        if is_labels:
            interpolater = sitk.sitkNearestNeighbor
            padding = 0
        else:
            interpolater = sitk.sitkLinear
            min_max_filter = sitk.MinimumMaximumImageFilter()
            min_max_filter.Execute(image)
            padding = min_max_filter.GetMinimum()
        
        rotated_image = sitk.Resample(image,
                                      transformation,
                                      interpolater,
                                      padding)
        
        return rotated_image, transformation
    
    
    def _random_rotate_z_axis(self, image: sitk.Image, is_labels: bool,
                              use_cache: bool = False) -> Tuple[sitk.Image, sitk.Euler3DTransform]:
        """
        Randomly rotates the input image around the z-axis within a specified
        range of degrees.

        Parameters
        ----------
        image : sitk.Image
            The input image to be rotated.
        is_labels : bool
            Indicates whether the image contains labels.
        use_cache : bool
            Indicates whether to use the cached rotation degrees or use the
            previous value, by default False.

        Returns
        -------
        rotated_image : sitk.Image
            The rotated image.
        transformation : sitk.Euler3DTransform
            The transformation matrix used to rotate the image.

        """
        if use_cache:
            degrees = self._cache_rotate_z_degrees
        else:
            degrees = self.random_generator.randint(self.min_z_rotation_degrees,
                                                    self.max_z_rotation_degrees,
                                                    size=None,
                                                    dtype=int)
            self._cache_rotate_z_degrees = degrees
        
        rotated_image, rotation_matrix = self._rotate_z_axis(image, degrees, is_labels=is_labels)
        
        return rotated_image, rotation_matrix

    
    @staticmethod
    def resample_image(image: sitk.Image, out_spacing: Tuple[float] = (1.0, 1.0, 1.0),
                       is_label: bool = False) -> sitk.Image:
        """
        Resamples the given image to a new spacing defined by the out_spacing
        parameter.
        
        Parameters
        ----------
            image : sitk.Image
                The input image to be resampled.
            out_spacing : Tuple[float], optional
                The desired spacing for the resampled image, by default
                (1.0, 1.0, 1.0)
            is_label : bool, optional
                Indicates whether the image is a label image, by default False
        
        Returns
        -------
        resampled_image : sitk.Image
            The resampled image.
        
        """
        original_spacing = np.array(image.GetSpacing())
        original_size = np.array(image.GetSize())
        
        if original_size[-1] == 1:
            out_spacing = list(out_spacing)
            out_spacing[-1] = original_spacing[-1]
            out_spacing = tuple(out_spacing)
    
        original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
        original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
        out_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)
    
        original_center = np.matmul(original_direction, original_center)
        out_center = np.matmul(original_direction, out_center)
        out_origin = np.array(image.GetOrigin()) + (original_center - out_center)
    
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(original_size.tolist())
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(out_origin.tolist())
        resample.SetTransform(sitk.Transform())
    
        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
            padding = 0
            resample.SetDefaultPixelValue(padding)
        else:
            resample.SetInterpolator(sitk.sitkLinear)
            min_max_filter = sitk.MinimumMaximumImageFilter()
            min_max_filter.Execute(image)
            padding = min_max_filter.GetMinimum()
            resample.SetDefaultPixelValue(padding)
    
        return resample.Execute(image)
    
    
    @staticmethod
    def _scale_image(image: sitk.Image, x_scale: float, y_scale: float,
                     z_scale: float, is_label: bool) -> sitk.Image:
        """
        Scale the given image by the specified scaling factors along each axis.
        
        Parameters
        ----------
        image :sitk.Image
            The input image to be scaled.
        x_scale : float
            Scaling factor along the x-axis.
        y_scale : float
            Scaling factor along the y-axis.
        z_scale : float
            Scaling factor along the z-axis.
        is_label : bool
            Indicates whether the image is a label image.
        
        Returns
        -------
        scaled_image : sitk.Image
            The scaled image.
        
        """
        spacing = np.asarray(image.GetSpacing())
        spacing[0] *= x_scale
        spacing[1] *= y_scale
        spacing[2] *= z_scale
        scaled_image = DataAugmentation.resample_image(image, spacing, is_label=is_label)

        return scaled_image
        
    
    def _random_image_scale(self, image: sitk.Image, is_labels: bool,
                            use_cache: bool = False) -> sitk.Image:
        """
        Randomly scales the given image anisotropically along each axis.

        Parameters
        ----------
        image : sitk.Image
            The input image to be scaled.
        is_labels : bool
            Indicates whether the image is a label image.
        use_cache : bool
            Indicates whether to use cached scaling factors from the previous
            execution, by default False.

        Returns
        -------
        scaled_image : sitk.Image
            The scaled image.

        """
        if use_cache:
            x_scale = self._cache_x_scale
            y_scale = self._cache_y_scale
            z_scale = self._cache_z_scale
        else:
            x_scale = self.random_generator.uniform(low=-self.max_abs_x_scale,
                                                    high=self.max_abs_x_scale)
            y_scale = self.random_generator.uniform(low=-self.max_abs_y_scale,
                                                    high=self.max_abs_y_scale)
            
            # Make sure bith x and y scales are both expanding or contracting
            if (x_scale < 0 and y_scale > 0) or (x_scale > 0 and y_scale < 0):
                y_scale *= -1
            
            x_scale += 1
            y_scale += 1
            
            z_scale = 1 + self.random_generator.uniform(low=-self.max_abs_z_scale,
                                                        high=self.max_abs_z_scale)
            
            self._cache_x_scale = x_scale
            self._cache_y_scale = y_scale
            self._cache_z_scale= z_scale
            
        scaled_image = self._scale_image(image, x_scale, y_scale, z_scale,
                                         is_label=is_labels)
        
        return scaled_image
    
        
    @staticmethod
    def _blur_image(image: sitk.Image, gaussian_sigma: float,
                    padding: Optional[float] = None) -> sitk.Image:
        """
        Blurs the given image using Gaussian blurring.

        Parameters
        ----------
        image : sitk.Image
            The input image to be blurred.
        gaussian_sigma : float
            The standard deviation of the Gaussian kernel used for blurring.
        padding : float, optional
           The padding value. If specified, the blur will be removed from
           the padded areas. The default is None.

        Returns
        -------
        blurred_image : sitk.Image
            The blurred image.

        """
        numpy_image = sitk.GetArrayFromImage(image)
        # In-plane only blurring
        blurred_image = ndimage.gaussian_filter(numpy_image, (gaussian_sigma,
                                                              gaussian_sigma,
                                                              0))
        
        if not padding is None:
            # Remove noise from padded areas
            blurred_image *= (numpy_image != 0)
            
        blurred_image = sitk.GetImageFromArray(blurred_image)
        blurred_image.CopyInformation(image)
        
        return blurred_image
    
    
    def _random_blur_image(self, image: sitk.Image, padding: Optional[float] = None,
                           use_cache: bool = False) -> sitk.Image:
        """
        Applies random Gaussian blurring to the given image.

        Parameters
        ----------
        image : sitk.Image
            The input image to be blurred.
        padding : float, optional
            The padding value. If specified, the blur will be removed from
            the padded areas. The default is None.
        use_cache : bool, optional
            Flag indicating whether to use cached blur sigma value from the
            previous execution, by default False.

        Returns
        -------
        image : sitk.Image
            The blurred image.

        """
        if use_cache:
            sigma = self._cache_blur_sigma
        else:
            sigma = self.random_generator.uniform(self.min_gaussian_blur_sigma,
                                                  self.max_gaussian_blur_sigma)
            
            self._cache_blur_sigma = sigma
            
        image = self._blur_image(image, sigma, padding)
        
        return image
    
        
    def _random_noise(self, image: sitk.Image, padding: Optional[float] = None) -> sitk.Image:
        """
        Adds random Gaussian noise to the given image.

        Parameters
        ----------
        image : sitk.Image
            The input image to add noise to.
        padding : float, optional
            The padding value. If specified, the random noise will be removed
            from the padded areas. The default is None.

        Returns
        -------
        noisy_image : sitk.Image
            The image with added Gaussian noise.

        """
        numpy_image = sitk.GetArrayFromImage(image)
        gaussian_noise = self.random_generator.normal(loc=0.0, scale=self.gaussian_scale,
                                                      size=numpy_image.shape)
        if not padding is None:
            # Remove noise from padded areas
            gaussian_noise *= (numpy_image != 0)
            
        numpy_image += gaussian_noise
        
        noisy_image = sitk.GetImageFromArray(numpy_image)
        noisy_image.CopyInformation(image)
        
        return noisy_image
    
    
    def _random_intensity_mean_shift(self, image: sitk.Image,
                                     padding: Optional[float] = None) -> sitk.Image:
        """
        Applies random intensity mean shift to the given image.

        Parameters
        ----------
        image : sitk.Image
            The input image to apply intensity mean shift to.
        padding : float, optional
            The padding value. If specified, the intensity shift field will be
            removed from the padded areas. The default is None.

        Returns
        -------
        intensity_image : sitk.Image
            The image with random intensity mean shift applied.

        """
        numpy_image = sitk.GetArrayFromImage(image)
        intensity_shift = self.random_generator.uniform(low=-self.max_intensity_scale,
                                                        high=self.max_intensity_scale)
        
        if not padding is None:
            # Remove noise from padded areas
            intensity_shift *= (numpy_image != 0)
        numpy_image += intensity_shift
        
        intensity_image = sitk.GetImageFromArray(numpy_image)
        intensity_image.CopyInformation(image)
        
        return intensity_image
    
    
    # Adapted from:
    # https://docs.monai.io/en/stable/_modules/monai/transforms/intensity/array.html#RandBiasField
    @staticmethod
    def _generate_bias_field(shape: List[int], degree: int, coeff: List[float]) -> np.ndarray:
        """
        Generates a bias field estimation using products of polynomials.

        Parameters
        ----------
        shape : List[int]
            The shape of the output bias field.
        degree : int
            The degree of the polynomials used for the bias field estimation.
        coeff : List[float]
            The coefficients of the polynomials.

        Raises
        ------
        NotImplementedError
            If the rank of the shape is not 2 or 3.

        Returns
        -------
        np.ndarray
            The bias field estimation.

        """
        """
        products of polynomials as bias field estimations
        """
        rank = len(shape)
        coeff_mat = np.zeros((degree + 1,) * rank)
        coords = [np.linspace(-1.0, 1.0, dim, dtype=np.float32) for dim in shape]
        if rank == 2:
            coeff_mat[np.tril_indices(degree + 1)] = coeff
            return np.polynomial.legendre.leggrid2d(coords[0], coords[1], coeff_mat)
        if rank == 3:
            pts = [[0, 0, 0]]
            for i in range(degree + 1):
                for j in range(degree + 1 - i):
                    for k in range(degree + 1 - i - j):
                        pts.append([i, j, k])
            if len(pts) > 1:
                pts = pts[1:]
            np_pts = np.stack(pts)
            coeff_mat[np_pts[:, 0], np_pts[:, 1], np_pts[:, 2]] = coeff
            return np.polynomial.legendre.leggrid3d(coords[0], coords[1], coords[2], coeff_mat)
        raise NotImplementedError("only supports 2D or 3D fields")
        
    
    def _random_bias_field(self, image: sitk.Image, padding: Optional[float] = None) -> sitk.Image:
        """
        Applies random bias field augmentation to the input image.

        Parameters
        ----------
        image : sitk.Image
            The input image to be augmented.
        padding : float, optional
            The padding value. If specified, the bias field will be removed from
            the padded areas. The default is None.

        Returns
        -------
        bias_field_image : sitk.Image
            The augmented image with random bias field.

        """
        numpy_image = sitk.GetArrayFromImage(image)
        
        n_coeff = int(np.prod([(self.degree + k) / k for k in range(1, len(numpy_image.shape) + 1)]))
        r_coeff = self.random_generator.uniform(*self.coeff_range, n_coeff).tolist()
        bias_field = self._generate_bias_field(numpy_image.shape, self.degree, r_coeff)
        
        bias_field = np.exp(bias_field)
        if not padding is None:
            # Remove noise from padded areas
            bias_field *= (numpy_image != 0)
        
        numpy_image = numpy_image * bias_field
        
        bias_field_image = sitk.GetImageFromArray(numpy_image)
        bias_field_image.CopyInformation(image)
        
        return bias_field_image
        
    
    # Adatped from:
    # https://github.com/dipy/dipy/blob/ecff0b9a88ea281bdbd2bcefd08d167ee79c301a/dipy/sims/voxel.py#L58
    @staticmethod
    def _generate_rician_noise(image: sitk.Image, noise1: np.ndarray, noise2: np.ndarray,
                               clip: bool = False, padding: Optional[float] = None) -> sitk.Image:
        """
        Generates Rician noise based on the input image and the provided noise arrays.

        Parameters
        ----------
        image : sitk.Image
            The input image.
        noise1 : np.ndarray
            The first noise array.
        noise2 : np.ndarray
            The second noise array.
        clip : bool, optional
            Flag indicating whether to clip the generated noise within the range of the input image. 
            The default is False.
        padding : float, optional
            The padding value. If specified, the noise will be removed from the
            padded areas. The default is None.

        Returns
        -------
        rician_noise : sitk.Image
            The image with Rician noise.

        """
        numpy_image = sitk.GetArrayFromImage(image)
    
        if clip:
            image_min = np.min(numpy_image)
            image_max = np.max(numpy_image)
        rician_noise = np.sqrt((numpy_image + noise1) ** 2 + noise2 ** 2)
        
        if clip:
            rician_noise = np.clip(rician_noise, a_min=image_min, a_max=image_max)
            
        if not padding is None:
            # Remove noise from padded areas
            rician_noise *= (numpy_image != 0)
            
        rician_noise = sitk.GetImageFromArray(rician_noise)
        rician_noise.CopyInformation(image)
        
        return rician_noise
        
    
    def _random_rician_noise(self, image: sitk.Image, padding: Optional[float] = None) -> sitk.Image:
        """
        Applies random Rician noise to the given image.

        Parameters
        ----------
        image : sitk.Image
            The input image to apply Rician noise to.
        padding : Optional[float], optional
            The padding value. If specified, the noise will be removed from the
            padded areas. The default is None.

        Returns
        -------
        rician_noise_image : sitk.Image
            The image with random Rician noise applied.

        """
        std = self.random_generator.uniform(0, self.rician_std)
        noise1 = self.random_generator.normal(self.rician_mean, std)
        noise2 = self.random_generator.normal(self.rician_mean, std)
        
        return self._generate_rician_noise(image, noise1, noise2,
                                           clip=True, padding=padding)

    
    def random_augmentation(self, image: sitk.Image, is_labels: bool,
                            likelihood: float, use_cache: bool = False) -> sitk.Image:
        """
        Applies a random augmentation to the given image.

        Parameters
        ----------
        image : sitk.Image
            The input image to apply augmentation to.
        is_labels : bool
            Flag indicating whether the image is a label image.
        likelihood : float
            The likelihood to apply the random augmentations. A value of 0
            indicates to disable augmentation anda value of 1 to always apply
            random augmentation.
        use_cache : bool, optional
            Flag indicating whether to cache randomly selected parameters for
            the transformations, by default False.

        Returns
        -------
        augmented_image : sitk.Image
            The image with random augmentation applied.

        """
        if use_cache:
            to_augment = self._cache_likelihood
        else:
            to_augment = self.random_generator.uniform(low=0.0, high=1.0) <= likelihood
            
            self._cache_likelihood = to_augment
        
        if not to_augment:
            return image
        
        image, _ = self._random_rotate_z_axis(image, is_labels, use_cache)
        image = self._random_image_scale(image, is_labels, use_cache)
        
        if not is_labels:
            image = self._random_rician_noise(image, padding=0)
            image = self._random_bias_field(image, padding=0)
        
        return image
        
    