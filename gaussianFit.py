import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from scipy.optimize import curve_fit
from matplotlib.widgets import RectangleSelector
def gaussian(x, x0, sigma, amplitude, offset):

	return np.exp(-(x-x0)**2/(2*sigma**2)) * amplitude + offset

def twoD_Gaussian((x,y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
		if np.abs(theta)>np.pi/4:
			theta = theta % np.pi/4
		xo = float(xo)
		yo = float(yo)    
		a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
		b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
		c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
		gauss2D = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
		return gauss2D.ravel() 


class fit2DGaussian(object):
	def __init__(self, image, diagnostics = False):
		self.im = image
		self.binnedImage = self.im
		
		print "Loaded Image"
		print "Image Shape: ", self.im.shape
		self.diagnostics = diagnostics
		
		
		xmax = self.im.shape[1]
		ymax = self.im.shape[0]
		
		self.ROI = [0, xmax, 0, ymax]
		
		self.bining = 1
		
		self.fitImage = self.im
		
		self.im /= self.im.max()
		
	def redefineROI(self):
		click = [None,None]
		release = [None,None]
		
		def RS_callback(eclick, erelease):
			click[:] = eclick.xdata, eclick.ydata
			release[:] = erelease.xdata, erelease.ydata
			
			for i in [0,1]:
				click[i] = int(click[i])
				release[i] = int(release[i])
			
			print "ROI Coords:", click, release
			return
			
		f, ax = plt.subplots(1,1)
		ax.imshow(self.binnedImage, origin='lower')
		RS = RectangleSelector(ax, RS_callback,
                                       drawtype='box', useblit=True,
                                       spancoords='pixels',
                                       interactive=True)
		plt.savefig('noROI.png')				   
		plt.show()
		ROI = [click[1], release[1], click[0], release[0]] # [y0, y1, x0, x1]
		print "ROI: ", ROI
		
		self.fitImage = self.binnedImage[ROI[0]:ROI[1],ROI[2]: ROI[3]]
		
		plt.imshow(self.fitImage, origin='lower')
		plt.savefig('ROI.png')
		plt.show()
		return
		
	def applyBinning(self, bining):	
		size_y = self.im.shape[0] / bining
		size_x = self.im.shape[1] / bining		
		
		image2D = self.im[:size_y*bining,:size_x*bining]
		
		shape = (size_y, bining, size_x, bining)
		
		binnedImage = image2D.reshape(shape).mean(-1).mean(1)
		
		binnedImage 
		
		self.binnedImage = binnedImage
		
		self.bining = bining
		return
	
	def fit1D(self,image1D):
		
		#Make an axis for the data
		size = image1D.shape[0]
		
		axis = np.linspace(0, size, size)
		
		#Guess at the correct values
		x0 = (axis * image1D).sum()/image1D.sum()
		amplitude = image1D.max()-image1D.min()
		offset = image1D.min()
		imageNoBG = image1D-offset
		sigma = np.sqrt(((axis - x0)**2*imageNoBG).sum()/imageNoBG.sum())
		
		guess = [x0, sigma, amplitude, offset]
		
		if self.diagnostics==True:
			#Show the 1D Guess and 1D data
			print '1D Guess: ', guess
			plt.scatter(axis, image1D)
			plt.plot(axis, gaussian(axis, *guess), 'r', label = "Guess")	
			plt.legend()
			plt.savefig('gaussian1.png')
			plt.show()
		
		
		p, cov = curve_fit(gaussian, axis, image1D, p0 = guess)
		
		if self.diagnostics==True:
			#Show the guess, result of the 1D fit and 1D data
			plt.scatter(axis, image1D)
			plt.plot(axis, gaussian(axis, *guess), 'r', label = "Guess")
			plt.plot(axis, gaussian(axis, *p), 'g', label = "fit")
			plt.legend()
			plt.savefig('gaussian1.png')
			plt.show()	
			
		
		return p

	def fit2D(self, image2D = None):
		if image2D == None:
			image2D = self.binnedImage
	
		xmax = image2D.shape[1]
		ymax = image2D.shape[0]
		
		x = np.linspace(0, xmax, xmax)
		y = np.linspace(0, ymax, ymax)
		
		x,y = np.meshgrid(x,y)
		
		
		image_x = np.average(image2D, axis = 0)
		image_y = np.average(image2D ,axis = 1)
		
		px = self.fit1D(image_x) # x0, sigma, amplitude, offset 
		py = self.fit1D(image_y)
		
		guess2D = [0,0,0,0,0,0,0] # amplitude, xo, yo, sigma_x, sigma_y, theta, offset
		guess2D[0] = (px[2]+py[2])/2
		guess2D[1] = px[0]
		guess2D[2] = py[0]
		guess2D[3] = px[1]
		guess2D[4] = py[1]
		guess2D[5] = 0
		guess2D[6] = (px[3]+py[3])/2
				
		shape = image2D.shape
		
		if self.diagnostics == True:
			#Show the 2D Guess and image with
			#binning and ROI.
			f, ax = plt.subplots(2,1)
			ax[0].imshow(image2D, origin='lower')
			ax[1].imshow(twoD_Gaussian((x,y), *guess2D).reshape(shape), origin='lower')
			plt.show()
		
		p2D, cov2D = curve_fit(twoD_Gaussian, (x,y), image2D.ravel(), p0 = guess2D)
		
		
		
		fit2D = twoD_Gaussian((x,y), *p2D).reshape(shape)
			
		if self.diagnostics == True:
			#Show the fit
			extent = [0, xmax, 0, ymax]
			
			f, ax = plt.subplots()
			
			ax.imshow(image2D, extent = extent, origin='lower')
			
			v = plt.axis()
			plt.contour(fit2D, extent = extent)
			plt.axis(v)
			
			plt.show()
		
		#Convert the values of x, y centers
		#and standard deviations to return
		#a measurement in units of the pixels
		#of the original image, not the binned image
		#used for the fit.
		
		print self.bining
		for i in [1,2,3,4]:
			p2D[i] *= self.bining
		
		return p2D
		

class loadAndFit(fit2DGaussian):
		def __init__(self, path, diagnostics = False):
			img = imread(path, 'F')[::-1]
			super(loadAndFit, self).__init__(img, diagnostics)