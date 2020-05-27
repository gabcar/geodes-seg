import SimpleITK as sitk
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

import os

# plt.ion()
plt.subplots(1, 4)


def dice(A, B):
    A = sitk.GetArrayFromImage(A)
    B = sitk.GetArrayFromImage(B)

    numer = 2 * np.sum(A * B)
    denom = np.sum(A + B)

    return numer / denom

path = '/mnt/Data/gabcar/visceral3/output/rbf_continuous_wb_sitk_capped_2/'

pats = os.listdir(path)

print(pats)

organs = [1, 2, 3, 4, 5, 6, 7]

for p in pats:
    print(p)

    pat_files = os.listdir(path + p)
    image = [f for f in pat_files if 'image' in f]
    mask = [f for f in pat_files if 'mask' in f]
    rbf_files = [f for f in pat_files if 'rbf3' in f]

    image = sitk.ReadImage(os.path.join(path, p, image[0]))
    gradientMagnitude = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    gradientMagnitude.SetSigma( 1 )

    # featureImage = sitk.BoundedReciprocal( gradientMagnitude.Execute( image ) ) # this ruins the input
    featureImage = gradientMagnitude.Execute( image ) 

    mask = sitk.ReadImage(os.path.join(path, p, mask[0]))

    for organ in organs:
        print(organ)
        rbf_organ = [f for f in rbf_files if 'organid{}'.format(organ) in f][0]

        rbf_sitk = sitk.ReadImage(os.path.join(path, p, rbf_organ))
        rbf = sitk.GetArrayFromImage(rbf_sitk)
        rbf = (rbf >= 0) * 1
        rbf = rbf.astype(np.uint8)

        # rbf = scipy.ndimage.morphology.binary_erosion(rbf, np.ones((3, 3, 3)), iterations=20).astype(np.uint8)

        rbf = sitk.GetImageFromArray(rbf)
        rbf.CopyInformation(rbf_sitk)
        rbf = sitk.Cast(rbf, sitk.sitkFloat32)

        # binary rbf mask
        initialImage = rbf

        # geo active contour filter
        geodesicActiveContour = sitk.GeodesicActiveContourLevelSetImageFilter()
        geodesicActiveContour.SetPropagationScaling( 0 ) # Balloon force
        geodesicActiveContour.SetCurvatureScaling( 1.0 ) # 
        geodesicActiveContour.SetAdvectionScaling( 1.0 )
        geodesicActiveContour.SetMaximumRMSError( 0.001 )
        geodesicActiveContour.SetNumberOfIterations( 1000 )

        # output = geo_seg(RBF, Edges of CT volume)
        out = geodesicActiveContour.Execute( initialImage, featureImage )

        print( "RMS Change: ", geodesicActiveContour.GetRMSChange() )
        print( "Elapsed Iterations: ", geodesicActiveContour.GetElapsedIterations() )

        plt.clf()
        plt.subplot(1, 4, 1)
        plt.title('initialImage')
        plt.imshow(np.max(sitk.GetArrayFromImage(initialImage), 1))
        plt.subplot(1, 4, 2)
        plt.title('featureImage')
        plt.imshow(np.mean(sitk.GetArrayFromImage(featureImage), 1))
        plt.subplot(1, 4, 3)
        plt.title('output')
        plt.imshow(np.max(sitk.GetArrayFromImage(out), 1))
        plt.subplot(1, 4, 4)
        plt.title('mask')
        plt.imshow(np.max(sitk.GetArrayFromImage(mask) == organ, 1))

        print('Dice mask vs rbf:', dice(mask == organ, rbf_sitk >= 0))
        print('Dice output vs mask:', dice(out >= 0, mask == organ))
        print('Dice output vs rbf:', dice(out >= 0, initialImage > 0))

        plt.show()

