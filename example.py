import gaussianFit

fitClass = gaussianFit.fit2DGaussian('./img.png', diagnostics = True)

params = fitClass.fit2D()

print params

#prints the list:
#[amplitude, xo, yo, sigma_x, sigma_y, theta, offset]