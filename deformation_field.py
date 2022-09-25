# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 08:09:54 2020
@author: lisicky

This program uses Image Registration - BSpline in default - to estimate deformation field
between two images. Dformation is aggregated if it is used to analyze deformation from e.g., 
mechanical testing. Subsequently, strain field can be calculated from the deformations. 
"""

import os
from typing import Optional
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import pandas as pd
import skimage.io as sk
import cv2
from scipy import ndimage
from tqdm import tqdm
import imageio
from scipy.io import savemat


class RegisterDeformations():
    def __init__(self, parameters, data):
        self.PATH_PARAMETERS = Path(parameters)
        self.PATH_DATA = Path(data)
        self.images = self.load_images()
        self.deformed_images = np.empty([self.images.shape[0], self.images.shape[1], self.images.shape[2]])
        self.strain_fields_XX = np.empty([self.images.shape[0] + 1, self.images.shape[1], self.images.shape[2]])
        self.strain_fields_YY = np.empty([self.images.shape[0] + 1, self.images.shape[1], self.images.shape[2]])
        self.strain_fields_XY = np.empty([self.images.shape[0] + 1, self.images.shape[1], self.images.shape[2]])
        self.strain_fields_XX[0,:,:] = np.zeros((self.images.shape[1], self.images.shape[2]))
        self.strain_fields_YY[0,:,:] = np.zeros((self.images.shape[1], self.images.shape[2]))
        self.strain_fields_XY[0,:,:] = np.zeros((self.images.shape[1], self.images.shape[2]))
        
        self.deformation_field_X = np.empty([self.images.shape[0] + 1, self.images.shape[1], self.images.shape[2]])
        self.deformation_field_Y = np.empty([self.images.shape[0] + 1, self.images.shape[1], self.images.shape[2]])
        self.deformation_field_X[0,:,:] = np.zeros((self.images.shape[1], self.images.shape[2]))
        self.deformation_field_Y[0,:,:] = np.zeros((self.images.shape[1], self.images.shape[2]))
       
    def crop_images(self, x0: int, x1: int, y0: int, y1: int):
        '''
        Images can be cropped to save some computation resources and to focus only to the object
        of interest.        

        Parameters
        ----------
        x0 : int
            DESCRIPTION. left boundary
        x1 : int
            DESCRIPTION. right boundary
        y0 : int
            DESCRIPTION. top boundary
        y1 : int
            DESCRIPTION. bottom boundary

        Returns
        -------
        None.
        '''

        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.images = self.images[:, self.y0:self.y1, self.x0:self.x1]
        self.deformation_field_X = self.deformation_field_X[:, self.y0:self.y1, self.x0:self.x1]
        self.deformation_field_Y = self.deformation_field_Y[:, self.y0:self.y1, self.x0:self.x1]
        self.strain_fields_XX = self.strain_fields_XX[:, self.y0:self.y1, self.x0:self.x1]
        self.strain_fields_YY = self.strain_fields_YY[:, self.y0:self.y1, self.x0:self.x1]
        self.strain_fields_XY = self.strain_fields_XY[:, self.y0:self.y1, self.x0:self.x1]
        self.deformed_images = self.deformed_images[:, self.y0:self.y1, self.x0:self.x1]
        
    def load_images(self) -> np.ndarray:
        '''
        Loads images which should be analyzed from a folder. Images should be in a separate folder.
        This method currently looks for .tif images only so other files can be also included in the
        folder.        

        Returns
        -------
        stack : np.ndarray
            DESCRIPTION. Input images in stack.
        '''

        images =  []
        for file in os.listdir(self.PATH_DATA):
            if file.endswith(".tif"):
                images.append(file)
        shape = sk.imread(self.PATH_DATA / images[0]).shape    # get shape of initial image
        stack = np.empty((len(images), shape[0], shape[1]))
        for i, image in enumerate(tqdm(images, desc="Loading images")):
            stack[i, :, :] = sk.imread(self.PATH_DATA / image)
        return stack
        
    def get_displacements(self) -> np.ndarray:
        '''
        Applies image registration with b-spline transformation to input images. Deformed image is
        taken as a fixed image and a image from state before is taken as moving image. Moving image
        is deformed to fit the fixed image in each iteration. Deformation is added to the previous
        state. ParameterMap which defines the process is taken from a file in the folder. If the 
        default options are not enought, then the parameters should be changed in the file.         

        Returns
        -------
        np.ndarray.
            DESRIPTION. Calculates deformation fields in x and y direction at each iteration.
        '''

        for i, image in enumerate(tqdm(self.images, desc="Processing images")):
            if i == self.images.shape[0] - 1:
                break
    
            fixed = self.images[i + 1, :, :]
            moving = image
    
            fixed_img = sitk.GetImageFromArray(fixed)
            moving_img = sitk.GetImageFromArray(moving)
            fixed_img.SetSpacing([1, 1, 1])
            moving_img.SetSpacing([1, 1, 1])
            fixed_img.SetOrigin([0,0,0])
            moving_img.SetOrigin([0,0,0])
    
            #  zahájení celé registrace
            elastixImageFilter = sitk.ElastixImageFilter()
            #elastixImageFilter.LogToFileOn()
            # nastavení snímků
            elastixImageFilter.SetFixedImage(fixed_img)
            elastixImageFilter.SetMovingImage(moving_img)
            # nastavení, které parametry chci mapovat. Nastavuji před samotnou registrací
            parameterMap = sitk.ReadParameterFile(str(self.PATH_PARAMETERS))
            elastixImageFilter.SetParameterMap(parameterMap)
            # run registration
            elastixImageFilter.Execute()
            # výsledek registrace a vypsání výsledků pro transformix!
            resultImage = elastixImageFilter.GetResultImage()
            resultIntImage = sitk.Cast(resultImage, sitk.sitkUInt8)
            self.deformed_images[i,:,:] = sitk.GetArrayFromImage(resultIntImage)
            transformParameterMap = elastixImageFilter.GetTransformParameterMap()
            transformixImageFilter = sitk.TransformixImageFilter()
            transformixImageFilter.LogToFileOn()
            transformixImageFilter.SetTransformParameterMap(transformParameterMap)
    
            x = np.arange(0, (moving).shape[1])
            y = np.arange(0, (moving).shape[0])
            X,Y = np.meshgrid(x,y)
            X = X + self.deformation_field_X[i,:,:]    # první deformation field bude O pro oba směry
            Y = Y + self.deformation_field_Y[i,:,:]
            X = list(X.ravel())
            Y= list(Y.ravel())
            points = []
            for j in range(len(X)):
                points.append("{} {}\n".format(X[j],Y[j]))
            
            with open("points.pts", "w") as f:
                f.write("%s\n" % "point")
                f.write("%s\n" % "{}".format(len(X)))
                for item in points:  
                    f.write("%s" % item)    
            
            transformixImageFilter.SetFixedPointSetFileName("points.pts")
            transformixImageFilter.Execute()
            
            deformation_field = self.process_points()
        
            x_dir = np.asarray(deformation_field["x_deformation"], dtype=float) * (-1)
            y_dir = np.asarray(deformation_field["y_deformation"], dtype=float) * (-1)
        
            grid_x = (x_dir.reshape((-1, fixed.shape[1]))).astype(float)
            grid_y = (y_dir.reshape((-1, fixed.shape[1]))).astype(float)
        
            self.deformation_field_X[i+1,:,:] = self.deformation_field_X[i,:,:] + grid_x
            self.deformation_field_Y[i+1,:,:] = self.deformation_field_Y[i,:,:] + grid_y
    
    def process_points(self) -> pd.DataFrame:
        '''
        Process points from transformix as the result is outputpoints.txt. Deformation in 
        x and y direction is calculated from the B-spline control points.        

        Returns
        -------
        pd.DataFrame
            DESCRIPTION. Dataframe with deformations for eac pixel in x and y direction.

        '''

        df = pd.read_csv("outputpoints.txt", delimiter=";", skiprows=0, header = None)
        df.columns = ["PointNo", "InputIndex", "InputPoint", "OutputIndexFixed", "OutputPoint", "Deformation"]
        deformation_field = pd.DataFrame()
        deformation_field[['str_inputPoints', "sep_InputPoints"]] = df['InputPoint'].str.split('=',expand=True)
        deformation_field[['str_deformation', "sep_deformation"]] = df['Deformation'].str.split('=',expand=True)
        deformation_field["InputPoints"] = deformation_field["sep_InputPoints"].apply(lambda x: x.replace(' [ ','').replace(']\t','')) 
        deformation_field["Deformation"] = deformation_field["sep_deformation"].apply(lambda x: x.replace(' [ ','').replace(']\t','')) 
        deformation_field[["x_input","y_input","nothing"]] = deformation_field["InputPoints"].str.split("\s", expand=True)
        deformation_field[["x_deformation","y_deformation","nothing2"]] = deformation_field["Deformation"].str.split("\s", expand=True)
        
        return deformation_field[["x_deformation", "y_deformation"]]
    
    def calc_strain(self, smooth: bool = False) -> np.ndarray:
        '''
        Calculates strains from the deformation fields. The smoothing is not needed as the deformation
        field is a result of B-splines which are smooth but it can be still applied. Default smoothing 
        is by the Gaussian filter.        

        Parameters
        ----------
        smooth : bool, optional
            DESCRIPTION. The default is False. Decide if smoothing of deformation field is required.

        Returns
        -------
        None.

        '''

        smoothed_XX = np.empty([self.deformation_field_X.shape[0], self.deformation_field_X.shape[1], self.deformation_field_X.shape[2]])
        smoothed_YY = np.empty([self.deformation_field_Y.shape[0], self.deformation_field_Y.shape[1], self.deformation_field_Y.shape[2]])
        
        for i, (dispX, dispY) in enumerate(tqdm(zip(self.deformation_field_X, self.deformation_field_Y)), total=len(self.deformaton_field_X)):
            if smooth:
                # vyhlazení posuvů kvůli šumu pomocí Gaussian filtru
                smoothed_XX[i,:,:] = ndimage.gaussian_filter(dispX, 5)
                smoothed_YY[i,:,:] = ndimage.gaussian_filter(dispY, 5)
                #smoothed_XX[i, :,:] = sgolay2d(dispX, window_size = 5, order = 1, derivative = None)
                #smoothed_YY[i, :,:] = sgolay2d(dispY, window_size = 5, order = 1, derivative = None)
            else:
                smoothed_XX[i,:,:] = dispX
                smoothed_YY[i,:,:] = dispY
            dudx = np.gradient(smoothed_XX[i,:,:], edge_order = 2, axis = 1)   
            dvdy = np.gradient(smoothed_YY[i,:,:], edge_order = 2, axis = 0)
            dudy = np.gradient(smoothed_XX[i,:,:], edge_order = 2, axis = 0)
            dvdx = np.gradient(smoothed_YY[i,:,:], edge_order = 2, axis = 1)
            self.strain_fields_XX[i+1,:,:] = 1/2*(2*dudx + (dudx)**2 + (dvdx)**2) 
            self.strain_fields_YY[i+1,:,:] = 1/2*(2*dvdy + (dudy)**2 + (dvdy)**2)
            self.strain_fields_XY[i+1,:,:] = 0.5*(dudy + dvdx + dudy*dudy + dvdx*dvdy)
    
    @staticmethod    
    def for_imageJ(image1: np.ndarray, image2: np.ndarray) -> None:
        '''
        Saves two images which can be than compared e.g., in ImageJ with stack animation or
        with colour overlay. This is much easier in ImageJ.

        Parameters
        ----------
        image1 : np.ndarray
            DESCRIPTION. This should be from original images -> self.images with indice i+1
            compared to image2 which is deformed image from image registration.
        image2 : np.ndarray
            DESCRIPTION. Deformed image from image registration -> self.deformed_images with
            indice i.

        Returns
        -------
        None
            DESCRIPTION. Saves two images for comparison.

        '''
        
        
        """ První obrázek je originál nahraný a měl by být i+1 oproti obrazu dva, který
            je zase zdeformovaný pomocí IR. Při kontrole v ImageJ originální deformace bych
            měl vždy kontrolovat právě image1 s obrazem o číslo menší. Jelikož i-1 deformuji."""
        cv2.imwrite("for_imageJ_0.tif", image1)
        cv2.imwrite("for_imageJ_1.tif", image2)
        
    def create_mask(self, characteristics: np.ndarray, limit: float) -> np.ndarray:
        '''
        This will put a characteristic into a mask which was created separatelz in ImageJ
        for the initial image. If there are local outliers which should be avoided in the 
        mask than limit can be used.

        Parameters
        ----------
        characteristics : np.ndarray
            DESCRIPTION. Slice of characteristic like deformation_field_X, strain_fields_XX, ...
        limit : float
            DESCRIPTION. It will be small number in case of strains and larger values for displacements.

        Returns
        -------
        masked : np.ndarray
            DESCRIPTION.
        '''

        mask = sk.imread("Mask.png")
        _, mask = cv2.threshold(mask[self.y0:self.y1, self.x0:self.x1], thresh=180, maxval=255, type=cv2.THRESH_BINARY_INV)
        masked = np.ma.masked_array(characteristics, ~mask)
        masked[masked > limit] = np.nan 
        return masked
    
    def plot_masked(self, masked: np.ndarray, name: str = None) -> None:
        '''
        Plots a masked characteristics on top of original image.

        Parameters
        ----------
        masked : np.ndarray
            DESCRIPTION. Masked characteristics like deformation_field_X, strain_fields_XX, ... Single
            image should be passed.
        name : str, optional
            DESCRIPTION. The default is None. If the name is passed, than the image will be also 
            saved for further use like e.g., creation of the gif.

        Returns
        -------
        None
        '''
    
        plt.figure()
        plt.imshow(self.images[0, :,:], cmap = "gray")    # default image
        im = plt.imshow(masked, alpha = 0.7, cmap = "jet")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(im)
        if name is not None:
            plt.savefig(name + ".tif")
        else:
            plt.show()
            
    def save_gif(self, characteristics: np.ndarray, save_name: str = None) -> None:
        '''
        Method which saves the images at each state of a specific characteristics and then creates
        a gif from it.        

        Parameters
        ----------
        characteristics : np.ndarray
            DESCRIPTION. This should be some of the characteristics like deformation_field_X, strain_fields_XX, ...
        save_name : str, optional
            DESCRIPTION. The default is None. If it should be a specific name of the gif than it should
            be specified here.
        ''' 
    
        images = []    # save image names for subsequent delete
        if save_name is not None:
            char_name = save_name
        else:
            char_name = "saved"
        with imageio.get_writer(f"{char_name}.gif", mode="I") as writer:
            for i, image in enumerate(tqdm(characteristics, desc="Processing masked images")):
                self.plot_masked(self.create_mask(image, 100), name = f"saved_image_{i}")
                images.append(f"saved_image_{i}.tif")
                image = imageio.imread(f"saved_image_{i}.tif")
                writer.append_data(image)
                path = Path(f"./saved_image_{i}.tif")
                path.unlink()

    @staticmethod
    def save_as_mat(data, name: Optional[str] = None) -> None:
        '''
        

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        '''
        if name is not None:
            dict_to_save = {f'{name}': data}
            savemat(f'{name}.mat', dict_to_save)
        else:
            dict_to_save = {'numpy.array': data}
            savemat(f'{name}.mat', dict_to_save)
            

if __name__ == '__main__':
    anal = RegisterDeformations(parameters='parameterMap.txt', data='./data/data_small')
    anal.crop_images(200, 600, 150, 600)
    anal.get_displacements()
    anal.save_gif(anal.deformation_field_Y, save_name='deformation_Y')
