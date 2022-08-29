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


def show_images(fixed, moving, title = None, dpi = 40, margin = 0.05):
    global ndf, ndm
    # ndf = sitk.GetArrayFromImage(fixed)
    # ndm = sitk.GetArrayFromImage(moving)
    spacing = pixelsize
    # print("Fixed image spacing is:", spacing)
    margin = 0.05
    figsize = (1 + margin) * fixed.shape[0] / dpi, (1 + margin) * fixed.shape[1] / dpi
    extent = (0, fixed.shape[1]*spacing, fixed.shape[0]*spacing, 0)
    figsize = (1 + margin) * fixed.shape[0] / dpi, (1 + margin) * fixed.shape[1] / dpi
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    plt.set_cmap("gray")
    ax.imshow(fixed, extent = extent,alpha=1, cmap="jet")
    ax.imshow(moving, extent = extent, alpha=0.5, cmap="Greens_r")
    


pixelsize = 1 
sliceID = 500
xRes = 800
yRes = 800
image_no = 2
path = Path("./data")

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
parameterMap["FinalGridSpacingInVoxels"] = ["128 128"]
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

parameterMap["MaximumStepLength"] = ["{}".format(pixelsize)]
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
parameterMap["Metric"] = [metrics[2]]
# počet binů histogramů pro určení mutual informace
parameterMap["NumberOfHistogramBins"] = ["64"]
parameterMap["NumberOfSpatialSamples"] = ["100000"] #Number of spatial samples used to compute the mutual information in each resolution level.
parameterMap["NewSamplesEveryIteration"] = ["true"]    # refresh těchto vzorků kažkou iteraci
parameterMap["ShowExactMetricValue"] = ["false"]
parameterMap[ "MaximumNumberOfSamplingAttempts" ] = ["4"]
# ____________________________ pyramidy __________________________________
parameterMap["FixedImagePyramid"] = [pyramids[0]]
parameterMap["MovingImagePyramid"] = [pyramids[1]]
parameterMap["NumberOfResolutions"] = ["3"]
parameterMap["FixedImagePyramidSchedule"] = ["8", "8", "4",  "4", "0", "0"]
parameterMap["MovingImagePyramidSchedule"] = ["8", "8", "4",  "4", "0", "0"]

parameterMap["WriteIterationInfo"] = ["true"]
parameterMap["ImageSampler"] = ["Random"]
parameterMap["WriteTransformParametersEachResolution"] = ["true"]

parameterMap["ResultImageFormat"] = ["raw"]
parameterMap["HowToCombineTransforms"] = ["Compose"]


  
# =============================================================================
# ========================= HLAVNÍ CYKLUS REGISTRACE ==========================
# =============================================================================
initial = path / "deformed_0.tif"
initial = sk.imread(initial)
initial=initial[260:550, 230:550]
# vytvoření hlavních matic, kam se mi budou ukládat výsledky z registračního procesu
deformed_images = np.empty([image_no, initial.shape[0], initial.shape[1]])
strain_fields_X = np.empty([image_no+1, initial.shape[0], initial.shape[1]])
strain_fields_Y = np.empty([image_no+1, initial.shape[0], initial.shape[1]])
strain_fields_X[0,:,:] = np.zeros((initial.shape[0], initial.shape[1]))
strain_fields_Y[0,:,:] = np.zeros((initial.shape[0], initial.shape[1]))
Deformation_field_X = np.empty([image_no+1, initial.shape[0], initial.shape[1]])
Deformation_field_Y = np.empty([image_no+1, initial.shape[0], initial.shape[1]])
Deformation_field_X[0,:,:] = np.zeros((initial.shape[0], initial.shape[1]))
Deformation_field_Y[0,:,:] = np.zeros((initial.shape[0], initial.shape[1]))

for i in range (0, image_no,1):
    fixed = path / "deformed_{}.tif".format(i+1)
    fixed_read = sk.imread(fixed)
    fixed_read=fixed_read[260:550, 230:550]
    
    moving = path / "deformed_{}.tif".format(i)
    moving_read = sk.imread(moving)
    moving_read=moving_read[260:550, 230:550]
    
    fixed_img = sitk.GetImageFromArray(fixed_read)
    moving_img = sitk.GetImageFromArray(moving_read)
    fixed_img.SetSpacing([pixelsize,pixelsize, pixelsize])
    moving_img.SetSpacing([pixelsize,pixelsize, pixelsize])
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
    x = np.arange(0, (moving_read).shape[1])
    y = np.arange(0, (moving_read).shape[0])
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

    x_dir = np.asarray(deformationField["x_deformation"], dtype=float)* (-1)
    y_dir = np.asarray(deformationField["y_deformation"], dtype=float) * (-1)

    Grid_x = (x_dir.reshape((-1, fixed_read.shape[1]))).astype(np.float)
    Grid_y = (y_dir.reshape((-1, fixed_read.shape[1]))).astype(np.float)

# zde by se mělo vytvořit pole posuvů, které bude stále narůstat a bude dále
# využíváno pro tvoření nových bodů v další iteraci
    Deformation_field_X[i+1,:,:] = Deformation_field_X[i,:,:] + Grid_x
    Deformation_field_Y[i+1,:,:] = Deformation_field_Y[i,:,:] + Grid_y
    
    Grid_x_1 = ndimage.gaussian_filter(Grid_x, 20)
    Grid_y_1 = ndimage.gaussian_filter(Grid_y, 20)
    
    strain_x = np.gradient(Grid_x_1, 1, axis = 1)
    strain_y = np.gradient(Grid_y_1, 1, axis = 0)

    strain_fields_X[i+1,:,:] = strain_fields_X[i,:,:] + strain_x
    strain_fields_Y[i+1,:,:] = strain_fields_Y[i,:,:] + strain_y
    print("One is done")


# for k in range(image_no+1):
#     fig, (ax,bx, cx) = plt.subplots(3,1, figsize=(5,10))
#     if k == 0:
#         ax.imshow(initial,cmap="gray")
#     else:
#         try:
#             ax.imshow(deformed_images[k,:,:], cmap="gray")
#         except:
#             pass    
        
#     crop = strain_fields_X.copy()   # jinak se to přepisuje!! Bída
#     crop  = crop[k,:,:]
#     crop[0:100, 0:crop.shape[1]] = np.nan    # pozor, zde je to už ořezaný kus, takže chce ladit na něj
#     crop[590:crop.shape[0], 0:crop.shape[1]] = np.nan
#     crop[0:crop.shape[0], 0:100]  = np.nan
#     crop[0:crop.shape[0], 540:crop.shape[1]] = np.nan                       
#     # fig = ax.imshow(strain_fields_X[3,215:390,518:830], cmap="jet" ,alpha = 1)
#     # fig = ax.imshow(crop, cmap="jet" ,alpha = 1)
    
#     bx.imshow(initial, alpha = 1, cmap="gray")
#     fig = bx.imshow(crop, cmap="jet" ,alpha = 0.8)
#     plt.colorbar(fig, ax = bx)

#     cx.set_xlim(0,30)
#     cx.set_ylim(0,8)
#     cx.set_xlabel("image no.")
#     cx.set_ylabel("Force F [N]")
#     cx.plot(np.linspace(0,k,k), forceX[0:k], label = "Applied force [N]")
#     plt.legend()
#     plt.subplots_adjust(bottom=0.15)
#     plt.savefig("strainX_{}".format(k+1))

# ======================== Kontrola procedené deformace =======================
img1 = path / "deformed_1.tif"    # zde musí být i+1 pro slazené obrazy. Pro zobrazení předchozího rozdílu je potřeba zobrazit i
img1 = sk.imread(img1)
img1=img1[260:550, 230:550]
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

