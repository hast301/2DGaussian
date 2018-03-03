import gaussianFit
import time
'''
#Make instance of the fit class around an image, with Diagnostics
fitClass = gaussianFit.fit2DGaussian('./img.png', diagnostics = True)

#Fit without bining

params = fitClass.fit2D()

print params

#prints the list:
#[amplitude, xo, yo, sigma_x, sigma_y, theta, offset]
'''

#Make instance of the fit class around an image, without diagnostics
fitClass = gaussianFit.fit2DGaussian('./img.png')

#Fit without bining

start = time.time()

params = fitClass.fit2D()

end = time.time()

print "Without Bining: ", end - start, "s"

print params

#Fit with bining

fitClass.applyBinning(3)

start = time.time()

params = fitClass.fit2D()

end = time.time()

print "With Bining: ", end - start, "s"

print params


