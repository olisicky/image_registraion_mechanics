# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 08:09:54 2020
@author: lisicky

This program uses Image Registration - BSpline in default - to estimate deformation field
between two images. Dformation is aggregated if it is used to analyze deformation from e.g., 
mechanical testing. Subsequently, strain field can be calculated from the deformations. 
"""

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import pandas as pd
import skimage.io as sk
from pathlib import Path
import cv2
import os
from scipy import ndimage
from tqdm import tqdm
import imageio


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
        arguments:
        x0 -- left boundary
        x1 -- right boundary
        y0 -- top boundary
        y1 -- bottom boundary
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
            elastixImageFilter.LogToFileOn()
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
        parameters:
        smooth -- boolean, decide if smoothing of deformation field is required
        '''
        smoothed_XX = np.empty([self.deformation_field_X.shape[0], self.Deformation_field_X.shape[1], self.Deformation_field_X.shape[2]])
        smoothed_YY = np.empty([self.deformation_field_Y.shape[0], self.Deformation_field_Y.shape[1], self.Deformation_field_Y.shape[2]])
        
        for dispX, dispY in zip(self.deformation_field_X, self.deformation_field_Y):
            if smooth:
                # vyhlazení posuvů kvůli šumu pomocí Gaussian filtru
                smoothed_XX[i,:,:] = ndimage.gaussian_filter(dispX, 10)
                smoothed_YY[i,:,:] = ndimage.gaussian_filter(dispY, 10)
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
    def for_imageJ(image1, image2):
        """ První obrázek je originál nahraný a měl by být i+1 oproti obrazu dva, který
            je zase zdeformovaný pomocí IR. Při kontrole v ImageJ originální deformace bych
            měl vždy kontrolovat právě image1 s obrazem o číslo menší. Jelikož i-1 deformuji."""
        cv2.imwrite("for_imageJ_0.tif", image1)
        cv2.imwrite("for_imageJ_1.tif", image2)
        
    def create_mask(self, characteristics, limit):
        """Tato funkce vykreslí výsledky zvolené veličiny na vzorku nedeformovaném,
            který je oříznut pomocí masky. Maska byla vytvořena pomocí imageJ. Limit
            je zde jenom z toho důvodu, že jsou tam místy lokální extrémy bodové, které
            rozhodí tu škálu. 
        """
        mask = sk.imread("Mask.png")
        _, mask = cv2.threshold(mask[self.y0:self.y1, self.x0:self.x1], thresh=180, maxval=255, type=cv2.THRESH_BINARY_INV)
        masked = np.ma.masked_array(characteristics, ~mask)
        masked[masked > limit] = np.nan 
        return masked
    
    def plot_masked(self, masked, name = None):
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
            
    def save_gif(self, characteristics): 
        images = []    # save image names for subsequent delete
        char_name = f'{characteristics}'.split('=')[0]
        with imageio.get_writer(f"{char_name}.gif", mode="I") as writer:
            for i, image in enumerate(tqdm(characteristics, desc="Processing masked images")):
                plot_masked(self.create_mask(image, 100), name = f"saved_image_{i}")
                images.append(f"saved_image_{i}.tif")
                image = imageio.imread(f"saved_image_{i}.tif")
                writer.append_data(image)
                path = Path(f"./saved_image_{i}.tif")
                path.unlink()

        
if __name__ == '__main__':
    anal = RegisterDeformations(parameters='parameterMap.txt', data='./data/data_small')
    anal.crop_images(200, 600, 150, 600)
    anal.get_displacements()
# =============================================================================
# PATH_PARAMETERS = Path("parameterMap.txt")
# PATH_DATA = Path("./data/data_small")
# 
# 
# parameterMap = sitk.ReadParameterFile(str(PATH_PARAMETERS))
#   
# # crop
# x0, x1, y0, y1 = 200, 600, 150, 600 
# 
# def load_images(path):
#     images =  []
#     for file in os.listdir(path):
#         if file.endswith(".tif"):
#             images.append(file)
#     shape = sk.imread(path / images[0]).shape    # get shape of initial image
#     stack = np.empty((len(images), shape[0], shape[1]))
#     for i, image in enumerate(tqdm(images, desc="Loading images")):
#         stack[i, :, :] = sk.imread(path / image)
#     return stack
# 
# 
# images = load_images(PATH_DATA)[:, y0:y1, x0:x1]
# 
# # vytvoření hlavních matic, kam se mi budou ukládat výsledky z registračního procesu
# deformed_images = np.empty([images.shape[0], images.shape[1], images.shape[2]])
# strain_fields_X = np.empty([images.shape[0] + 1, images.shape[1], images.shape[2]])
# strain_fields_Y = np.empty([images.shape[0] + 1, images.shape[1], images.shape[2]])
# strain_fields_X[0,:,:] = np.zeros((images.shape[1], images.shape[2]))
# strain_fields_Y[0,:,:] = np.zeros((images.shape[1], images.shape[2]))
# Deformation_field_X = np.empty([images.shape[0] + 1, images.shape[1], images.shape[2]])
# Deformation_field_Y = np.empty([images.shape[0] + 1, images.shape[1], images.shape[2]])
# Deformation_field_X[0,:,:] = np.zeros((images.shape[1], images.shape[2]))
# Deformation_field_Y[0,:,:] = np.zeros((images.shape[1], images.shape[2]))
# 
# for i, image in enumerate(tqdm(images, desc="Processing images")):
#     if i == images.shape[0] - 1:
#         break
#     
#     fixed = images[i + 1, :, :]
#     moving = image
#     
#     fixed_img = sitk.GetImageFromArray(fixed)
#     moving_img = sitk.GetImageFromArray(moving)
#     fixed_img.SetSpacing([1, 1, 1])
#     moving_img.SetSpacing([1, 1, 1])
#     fixed_img.SetOrigin([0,0,0])
#     moving_img.SetOrigin([0,0,0])
#     
#     #  zahájení celé registrace
#     elastixImageFilter = sitk.ElastixImageFilter()
#     elastixImageFilter.LogToFileOn()
#     # nastavení snímků
#     elastixImageFilter.SetFixedImage(fixed_img)
#     elastixImageFilter.SetMovingImage(moving_img)
#     # nastavení, které parametry chci mapovat. Nastavuji před samotnou registrací
#     elastixImageFilter.SetParameterMap(parameterMap)
#     # spuštěšní procesu registrace
#     elastixImageFilter.Execute()
#     # výsledek registrace a vypsání výsledků pro transformix!
#     resultImage = elastixImageFilter.GetResultImage()
#     resultIntImage = sitk.Cast(resultImage, sitk.sitkUInt8)
#     deformed_images[i,:,:] = sitk.GetArrayFromImage(resultIntImage)
#     transformParameterMap = elastixImageFilter.GetTransformParameterMap()
#     transformixImageFilter = sitk.TransformixImageFilter()
#     transformixImageFilter.LogToFileOn()
#     transformixImageFilter.SetTransformParameterMap(transformParameterMap)
#     
#     x = np.arange(0, (moving).shape[1])
#     y = np.arange(0, (moving).shape[0])
#     X,Y = np.meshgrid(x,y)
#     X = X + Deformation_field_X[i,:,:]    # první deformation field bude O pro oba směry
#     Y = Y + Deformation_field_Y[i,:,:]
#     X = list(X.ravel())
#     Y= list(Y.ravel())
#     points = []
#     for j in range(len(X)):
#         points.append("{} {}\n".format(X[j],Y[j]))
#     
#     with open("points.pts", "w") as f:
#         f.write("%s\n" % "point")
#         f.write("%s\n" % "{}".format(len(X)))
#         for item in points:  
#             f.write("%s" % item)    
#     
#     transformixImageFilter.SetFixedPointSetFileName("points.pts")
#     transformixImageFilter.Execute()
#     df = pd.read_csv("outputpoints.txt", delimiter=";", skiprows=0, header = None)
#     df.columns = ["PointNo", "InputIndex", "InputPoint", "OutputIndexFixed", "OutputPoint", "Deformation"]
#     deformationField = pd.DataFrame()
#     deformationField[['str_inputPoints', "sep_InputPoints"]] = df['InputPoint'].str.split('=',expand=True)
#     deformationField[['str_deformation', "sep_deformation"]] = df['Deformation'].str.split('=',expand=True)
#     deformationField["InputPoints"] = deformationField["sep_InputPoints"].apply(lambda x: x.replace(' [ ','').replace(']\t','')) 
#     deformationField["Deformation"] = deformationField["sep_deformation"].apply(lambda x: x.replace(' [ ','').replace(']\t','')) 
#     deformationField[["x_input","y_input","nothing"]] = deformationField["InputPoints"].str.split("\s", expand=True)
#     deformationField[["x_deformation","y_deformation","nothing2"]] = deformationField["Deformation"].str.split("\s", expand=True)
# 
#     x_dir = np.asarray(deformationField["x_deformation"], dtype=float) * (-1)
#     y_dir = np.asarray(deformationField["y_deformation"], dtype=float) * (-1)
# 
#     Grid_x = (x_dir.reshape((-1, fixed.shape[1]))).astype(float)
#     Grid_y = (y_dir.reshape((-1, fixed.shape[1]))).astype(float)
# 
#     Deformation_field_X[i+1,:,:] = Deformation_field_X[i,:,:] + Grid_x
#     Deformation_field_Y[i+1,:,:] = Deformation_field_Y[i,:,:] + Grid_y
#     
# 
# def calc_strain(displacement_X, displacement_Y):
#     # nejprve vyhlazení posuvů
#     smoothed_XX = np.empty([displacement_X.shape[0], displacement_X.shape[1], displacement_X.shape[2]])
#     smoothed_YY = np.empty([displacement_Y.shape[0], displacement_Y.shape[1], displacement_Y.shape[2]])
#     E_xx = np.empty([displacement_X.shape[0], displacement_X.shape[1], displacement_X.shape[2]])
#     E_yy = np.empty([displacement_Y.shape[0], displacement_Y.shape[1], displacement_Y.shape[2]])
#     E_xy = np.empty([displacement_Y.shape[0], displacement_Y.shape[1], displacement_Y.shape[2]])
# 
#     i = 0
#     for dispX, dispY in zip(displacement_X, displacement_Y):
#             # vyhlazení posuvů kvůli šumu pomocí Gaussian filtru
#         #smoothed_XX[i,:,:] = ndimage.gaussian_filter(disp_x, 10)
#         #smoothed_YY[i,:,:] = ndimage.gaussian_filter(disp_y, 10)
#         #smoothed_XX[i, :,:] = sgolay2d(dispX, window_size = 5, order = 1, derivative = None)
#         #smoothed_YY[i, :,:] = sgolay2d(dispY, window_size = 5, order = 1, derivative = None)
#         smoothed_XX[i,:,:] = dispX
#         smoothed_YY[i,:,:] = dispY
#         dudx = np.gradient(smoothed_XX[i,:,:], edge_order = 2, axis = 1)   
#         dvdy = np.gradient(smoothed_YY[i,:,:], edge_order = 2, axis = 0)
#         dudy = np.gradient(smoothed_XX[i,:,:], edge_order = 2, axis = 0)
#         dvdx = np.gradient(smoothed_YY[i,:,:], edge_order = 2, axis = 1)
#         E_xx[i,:,:] = 1/2*(2*dudx + (dudx)**2 + (dvdx)**2) 
#         E_yy[i,:,:] = 1/2*(2*dvdy + (dudy)**2 + (dvdy)**2)
#         E_xy[i,:,:] = 0.5*(dudy + dvdx + dudy*dudy + dvdx*dvdy)
#         i = i+1
#     return E_xx, E_yy, E_xy
# 
# 
# E_xx, E_yy, E_xy = calc_strain(Deformation_field_X, Deformation_field_Y) 
# 
# def for_imageJ(image1, image2):
#     """ První obrázek je originál nahraný a měl by být i+1 oproti obrazu dva, který
#         je zase zdeformovaný pomocí IR. Při kontrole v ImageJ originální deformace bych
#         měl vždy kontrolovat právě image1 s obrazem o číslo menší. Jelikož i-1 deformuji."""
#     cv2.imwrite("for_imageJ_0.tif", image1)
#     cv2.imwrite("for_imageJ_1.tif", image2)
# 
# 
# for_imageJ(images[3, :,:], deformed_images[2,:,:])
# 
# 
# def create_mask(characteristics, limit):
#     """"Tato funkce vykreslí výsledky zvolené veličiny na vzorku nedeformovaném,
#         který je oříznut pomocí masky. Maska byla vytvořena pomocí imageJ. Limit
#         je zde jenom z toho důvodu, že jsou tam místy lokální extrémy bodové, které
#         rozhodí tu škálu. """
#     mask = sk.imread("Mask.png")
#     _, mask = cv2.threshold(mask[y0:y1, x0:x1], thresh=180, maxval=255, type=cv2.THRESH_BINARY_INV)
#     masked = np.ma.masked_array(characteristics, ~mask)
#     masked[masked > limit] = np.nan 
#     return masked
# 
# 
# def plot_masked(masked, name = None):
#     plt.figure()
#     plt.imshow(images[0, :,:], cmap = "gray")    # default image
#     im = plt.imshow(masked, alpha = 0.7, cmap = "jet")
#     plt.xticks([])
#     plt.yticks([])
#     plt.colorbar(im)
#     if name is not None:
#         plt.savefig(name + ".tif")
#     else:
#         plt.show()
#  
#     
# mask = plot_masked(create_mask(Deformation_field_X[3,:,:], 0.8))
# 
# 
# def save_gif(characteristics): 
#     images = []    # save image names for subsequent delete
#     with imageio.get_writer("smiling.gif", mode="I") as writer:
#         for i, image in enumerate(tqdm(characteristics, desc="Processing masked images")):
#             plot_masked(create_mask(image, 100), name = f"saved_image_{i}")
#             images.append(f"saved_image_{i}.tif")
#             image = imageio.imread(f"saved_image_{i}.tif")
#             writer.append_data(image)
#             path = Path(f"./saved_image_{i}.tif")
#             path.unlink()
#     
# save_gif(Deformation_field_X)
# =============================================================================
