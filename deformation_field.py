# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 08:09:54 2020
@author: lisicky

Rev_2 by měla brát v potaz více snímků najednou pro podchycení celé chrakateristiky
"""

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib.gridspec as gridspec
import skimage.io as sk
from scipy import ndimage
from pathlib import Path
import cv2
import os


path = Path("./data/data_small")

# =============================================================================
# ========================== IMAGE REGISTRATION ===============================
# =============================================================================
# Převedení array obrazu do sitk image pro další zpracování
# fixed_img = sitk.GetImageFromArray(fixedImgOrig)
# moving_img = sitk.GetImageFromArray(movingImgOrig)

metrics = ["AdvancedMeanSquares", "AdvancedNormalizedCorrelation", 
           "AdvancedMattesMutualInformation", "NormalizedMutualInformation",
           "AdvancedKappaStatistic"]
samplers = ["Full", "Random", "Grid", "RandomCoordinate"]
interpolates = ["NearestNeighborInterpolator", "LinearInterpolator", 
                "BSplineInterpolator", "BSplineInterpolatorFloat"]
transformation=["BSplineTransform", "EulerTransform", "TranslationTransform",
                "SimilarityTransform", "AffineTransform",
                "SplineKernelTransform", "DeformationFieldTransform"]
optimisers = ["QuasiNewtonLBFGS", "RegularStepGradientDescent", 
              "FiniteDifferenceGradientDescent",
              "AdaptiveStochasticGradientDescent", "QuasiNewtonLBFGS", "RSGDEachParameterApart", "SimultaneousP"]
pyramids = ["FixedRecursiveImagePyramid", "MovingRecursiveImagePyramid", "FixedSmoothingImagePyramid",
            "MovingSmoothingImagePyramid", "FixedShrinkingImagePyramid", 
            "MovingShrinkingImagePyramid"]
registration = ["MultiResolutionRegistration", "MultiMetricMultiResolutionRegistration", 
                "MultiResolutionRegistrationWithFeatures", "RegistrationBase"]
# parameterMap = sitk.GetDefaultParameterMap('rigid')
parameterMap=sitk.ParameterMap()
parameterMap["Registration"] = [registration[0]]
parameterMap["UseDirectionCosines"] = ["true"]
parameterMap['Transform'] = [transformation[0]]
parameterMap["FinalGridSpacingInVoxels"] = ["64 64"]
parameterMap["GridSpacingSchedule"] =  ["1", "1", "1"]
outValue = 0
parameterMap["DefaultPixelValue"] = ["{}".format(outValue)]   # default pixel value of pixels which come out of picture
# _______________________ základní informace o snímcích _____________________
parameterMap["FixedImageDimension"] = ["2"]
parameterMap["MovingImageDimension"] = ["2"]
# _________________________ optimisers ___________________________________
parameterMap["Optimizer"] = [optimisers[3]]
# podstatné zvolit, když využívám adaptiveStochasticGradientDescent!!
parameterMap["AutomaticParameterEstimation"] = ["true"]
parameterMap[ "MaximumNumberOfSamplingAttempts" ] = ["2048"]
parameterMap["NumberOfGradientMeasurements"] = ["10"]
parameterMap["NumberOfJacobianMeasurements"] = ["100000"]

parameterMap["MaximumStepLength"] = ["1"]
parameterMap['MaximumNumberOfIterations'] = ["500", "500", "1000"]

# Quasi Newton
# parameterMap["StepLength"] = ["0.0005", "0.0005", "0.0005"]  #StepLength: Set the length of the initial step
# parameterMap["MaximumStepLength"] = ["0.0001", "0.0001", "0.0001"]
# parameterMap["LBFGSUpdateAccuracy"] = ["10", "20","60"]  #The "memory" of the optimizer. This determines how many past iterations are used to construct the Hessian approximation. The higher, the more memory is used, but the better the Hessian approximation
# parameterMap["StopIfWolfeNotSatisfied"] = ["false"]
# __________________________ interpolace _________________________________
parameterMap["Interpolator"] = [interpolates[2]]
parameterMap["BSplineInterpolationOrder"] = ["3"]
# ___________________________ metrika ______________________________________
parameterMap["Metric"] = [metrics[1]]
# počet binů histogramů pro určení mutual informace
parameterMap["NumberOfHistogramBins"] = ["16"]
parameterMap["NumberOfSpatialSamples"] = ["400000"] #Number of spatial samples used to compute the mutual information in each resolution level.
parameterMap["NewSamplesEveryIteration"] = ["true"]    # refresh těchto vzorků kažkou iteraci
parameterMap["ShowExactMetricValue"] = ["false"]
parameterMap[ "MaximumNumberOfSamplingAttempts" ] = ["4"]
# ____________________________ pyramidy __________________________________
parameterMap["FixedImagePyramid"] = [pyramids[2]]
parameterMap["MovingImagePyramid"] = [pyramids[3]]
parameterMap["NumberOfResolutions"] = ["3"]
parameterMap["FixedImagePyramidSchedule"] = ["4", "4", "2",  "2", "0", "0"]
parameterMap["MovingImagePyramidSchedule"] = ["4", "4", "2",  "2", "0", "0"]

parameterMap["WriteIterationInfo"] = ["true"]
parameterMap["ImageSampler"] = ["Random"]
parameterMap["WriteTransformParametersEachResolution"] = ["true"]

parameterMap["ResultImageFormat"] = ["raw"]
parameterMap["HowToCombineTransforms"] = ["Compose"]


  
# =============================================================================
# ========================= HLAVNÍ CYKLUS REGISTRACE ==========================
# =============================================================================
x0, x1, y0, y1 = 200, 600, 150, 600 

def load_images(path):
    images =  []
    for file in os.listdir(path):
        if file.endswith(".tif"):
            images.append(file)
    print(images)
    shape = sk.imread(path / images[0]).shape    # get shape of initial image
    stack = np.empty((len(images), shape[0], shape[1]))
    for i, image in enumerate(images):
        stack[i, :, :] = sk.imread(path / image)
    return stack

images = load_images(path)[:, y0:y1, x0:x1]

# vytvoření hlavních matic, kam se mi budou ukládat výsledky z registračního procesu
deformed_images = np.empty([images.shape[0], images.shape[1], images.shape[2]])
strain_fields_X = np.empty([images.shape[0] + 1, images.shape[1], images.shape[2]])
strain_fields_Y = np.empty([images.shape[0] + 1, images.shape[1], images.shape[2]])
strain_fields_X[0,:,:] = np.zeros((images.shape[1], images.shape[2]))
strain_fields_Y[0,:,:] = np.zeros((images.shape[1], images.shape[2]))
Deformation_field_X = np.empty([images.shape[0] + 1, images.shape[1], images.shape[2]])
Deformation_field_Y = np.empty([images.shape[0] + 1, images.shape[1], images.shape[2]])
Deformation_field_X[0,:,:] = np.zeros((images.shape[1], images.shape[2]))
Deformation_field_Y[0,:,:] = np.zeros((images.shape[1], images.shape[2]))

for i, image in enumerate(images):
    if i == images.shape[0] - 1:
        break
    
    fixed = images[i + 1, :, :]
    moving = image
    
    fixed_img = sitk.GetImageFromArray(fixed)
    moving_img = sitk.GetImageFromArray(moving)
    fixed_img.SetSpacing([1, 1, 1])
    moving_img.SetSpacing([1, 1, 1])
    fixed_img.SetOrigin([0,0,0])
    moving_img.SetOrigin([0,0,0])
    
    #  zahájení celé registrace
    elastixImageFilter = sitk.ElastixImageFilter()
    # vypsání základních parametrů. Musí to být přes sitk, protože vypsání od elastixu má
    # problém v některých IDLE
    sitk.WriteParameterFile(parameterMap, filename="parameterMap.txt")
    elastixImageFilter.LogToFileOn()
    # nastavení snímků
    elastixImageFilter.SetFixedImage(fixed_img)
    elastixImageFilter.SetMovingImage(moving_img)
    # nastavení, které parametry chci mapovat. Nastavuji před samotnou registrací
    elastixImageFilter.SetParameterMap(parameterMap)
    # spuštěšní procesu registrace
    elastixImageFilter.Execute()
    # výsledek registrace a vypsání výsledků pro transformix!
    resultImage = elastixImageFilter.GetResultImage()
    resultIntImage = sitk.Cast(resultImage, sitk.sitkUInt8)
    deformed_images[i,:,:] = sitk.GetArrayFromImage(resultIntImage)
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.LogToFileOn()
    transformixImageFilter.SetTransformParameterMap(transformParameterMap)
    
# =============================================================================
# ===================== TVORBA BODŮ PRO TRANSFORMACI ==========================
# =============================================================================    
    x = np.arange(0, (moving).shape[1])
    y = np.arange(0, (moving).shape[0])
    X,Y = np.meshgrid(x,y)
    X = X + Deformation_field_X[i,:,:]    # první deformation field bude O pro oba směry
    Y = Y + Deformation_field_Y[i,:,:]
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
    df = pd.read_csv("outputpoints.txt", delimiter=";", skiprows=0, header = None)

    # =============================================================================
    # ===================== TRANSFORMACE BODŮ MŘÍŽKY =============================
    # =============================================================================
    
    
    df = pd.read_csv("outputpoints.txt", delimiter=";", skiprows=0, header = None)
    df.columns = ["PointNo", "InputIndex", "InputPoint", "OutputIndexFixed", "OutputPoint", "Deformation"]
    deformationField = pd.DataFrame()
    deformationField[['str_inputPoints', "sep_InputPoints"]] = df['InputPoint'].str.split('=',expand=True)
    deformationField[['str_deformation', "sep_deformation"]] = df['Deformation'].str.split('=',expand=True)
    deformationField["InputPoints"] = deformationField["sep_InputPoints"].apply(lambda x: x.replace(' [ ','').replace(']\t','')) 
    deformationField["Deformation"] = deformationField["sep_deformation"].apply(lambda x: x.replace(' [ ','').replace(']\t','')) 
    deformationField[["x_input","y_input","nothing"]] = deformationField["InputPoints"].str.split("\s", expand=True)
    deformationField[["x_deformation","y_deformation","nothing2"]] = deformationField["Deformation"].str.split("\s", expand=True)

# ======================= Odečtení dat po registraci ==========================

    x_dir = np.asarray(deformationField["x_deformation"], dtype=float) * (-1)
    y_dir = np.asarray(deformationField["y_deformation"], dtype=float) * (-1)

    Grid_x = (x_dir.reshape((-1, fixed.shape[1]))).astype(np.float)
    Grid_y = (y_dir.reshape((-1, fixed.shape[1]))).astype(np.float)

# zde by se mělo vytvořit pole posuvů, které bude stále narůstat a bude dále
# využíváno pro tvoření nových bodů v další iteraci
    Deformation_field_X[i+1,:,:] = Deformation_field_X[i,:,:] + Grid_x
    Deformation_field_Y[i+1,:,:] = Deformation_field_Y[i,:,:] + Grid_y
    
    print(f'{i}th image is done')

def calc_strain(displacement_X, displacement_Y):
    # nejprve vyhlazení posuvů
    smoothed_XX = np.empty([displacement_X.shape[0], displacement_X.shape[1], displacement_X.shape[2]])
    smoothed_YY = np.empty([displacement_Y.shape[0], displacement_Y.shape[1], displacement_Y.shape[2]])
    E_xx = np.empty([displacement_X.shape[0], displacement_X.shape[1], displacement_X.shape[2]])
    E_yy = np.empty([displacement_Y.shape[0], displacement_Y.shape[1], displacement_Y.shape[2]])
    E_xy = np.empty([displacement_Y.shape[0], displacement_Y.shape[1], displacement_Y.shape[2]])

    i = 0
    for dispX, dispY in zip(displacement_X, displacement_Y):
            # vyhlazení posuvů kvůli šumu pomocí Gaussian filtru
        #smoothed_XX[i,:,:] = ndimage.gaussian_filter(disp_x, 10)
        #smoothed_YY[i,:,:] = ndimage.gaussian_filter(disp_y, 10)
        #smoothed_XX[i, :,:] = sgolay2d(dispX, window_size = 5, order = 1, derivative = None)
        #smoothed_YY[i, :,:] = sgolay2d(dispY, window_size = 5, order = 1, derivative = None)
        smoothed_XX[i,:,:] = dispX
        smoothed_YY[i,:,:] = dispY
        dudx = np.gradient(smoothed_XX[i,:,:], edge_order = 2, axis = 1)   
        dvdy = np.gradient(smoothed_YY[i,:,:], edge_order = 2, axis = 0)
        dudy = np.gradient(smoothed_XX[i,:,:], edge_order = 2, axis = 0)
        dvdx = np.gradient(smoothed_YY[i,:,:], edge_order = 2, axis = 1)
        E_xx[i,:,:] = 1/2*(2*dudx + (dudx)**2 + (dvdx)**2) 
        E_yy[i,:,:] = 1/2*(2*dvdy + (dudy)**2 + (dvdy)**2)
        E_xy[i,:,:] = 0.5*(dudy + dvdx + dudy*dudy + dvdx*dvdy)
        i = i+1
    return E_xx, E_yy, E_xy


E_xx, E_yy, E_xy = calc_strain(Deformation_field_X, Deformation_field_Y) 



def for_imageJ(image1, image2):
    """ První obrázek je originál nahraný a měl by být i+1 oproti obrazu dva, který
        je zase zdeformovaný pomocí IR. Při kontrole v ImageJ originální deformace bych
        měl vždy kontrolovat právě image1 s obrazem o číslo menší. Jelikož i-1 deformuji."""
    #img1 = image1    # zde musí být i+1 pro slazené obrazy. Pro zobrazení předchozího rozdílu je potřeba zobrazit i
    #img1 = sk.imread(img1)
    #img1=img1[y0:y1, x0:x1]
    #img2 = image2    # zde musí být i+1 pro slazené obrazy. Pro zobrazení předchozího rozdílu je potřeba zobrazit i
    #img2 = sk.imread(img2)
    #img2=img2[y0:y1, x0:x1]
    
    cv2.imwrite("for_imageJ_0.tif", image1)
    cv2.imwrite("for_imageJ_1.tif", image2)
    #cv2.imwrite("for_imageJ_1.tif", img2)
    
for_imageJ(images[1, :,:], deformed_images[0,:,:])


def plot_sample(characteristics, limit):
    """"Tato funkce vykreslí výsledky zvolené veličiny na vzorku nedeformovaném,
        který je oříznut pomocí masky. Maska byla vytvořena pomocí imageJ. Limit
        je zde jenom z toho důvodu, že jsou tam místy lokální extrémy bodové, které
        rozhodí tu škálu. """
    mask = sk.imread("Mask.png")
    _, mask = cv2.threshold(mask[y0:y1, x0:x1], thresh=180, maxval=255, type=cv2.THRESH_BINARY_INV)
    masked = np.ma.masked_array(characteristics, ~mask)
    masked[masked > limit] = np.nan 
    plt.figure()
    plt.imshow(images[0, :,:], cmap = "gray")
    im = plt.imshow(masked, alpha = 0.7, cmap = "jet")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im)
    return masked

mask = plot_sample(E_yy[6,:,:], 0.8)



# ======================== Kontrola procedené deformace =======================
img1 = path / "deformed_1.tif"    # zde musí být i+1 pro slazené obrazy. Pro zobrazení předchozího rozdílu je potřeba zobrazit i
img1 = sk.imread(img1)
img1=img1[x0:x1, y0:y1]
img2 = deformed_images[1,:,:]   # zde je i
show_images(img2, img1)



# img1 = sk.imread(img1)
# img1 = img1[250:800, 0:xRes]    
# cmap = plt.cm.Greens_rů
# norm = plt.Normalize(vmin=img1.min(), vmax=img1.max())
# image = cmap(norm(img1))
# # save the image
# plt.imsave('test.png', image)


# cmap = plt.cm.Purples_r
# norm = plt.Normalize(vmin=img1.min(), vmax=img1.max())
# image1 = cmap(norm(img2))
# # save the image
# plt.imsave('test1.png', image1)

# img3 = Image.open("./test.png")
# img4 = Image.open("./test1.png")
# mask = Image.new("L", img3.size, color=100)
# Image.composite(img3,img4, mask)
# resized = cv2.resize(moving_read, (int((GridSize[0])), int(GridSize[1])), interpolation = cv2.INTER_NEAREST)

