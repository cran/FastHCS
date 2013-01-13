\name{FastHCS-package}
\docType{package}
\title{Package implementing the FastHCS robust PCA algorithm.}
\description{
Uses the FastHCS algorithm to compute a robust PCA model.
}
\details{
\tabular{ll}{
Package: \tab FastHCS\cr
Type: \tab Package\cr
Version: \tab 0.1\cr
Date: \tab 2013-01-13\cr
Suggests: \tab mvtnorm\cr
License: \tab GPL (>= 2)\cr
LazyLoad: \tab yes\cr
}

Index:
\preformatted{
FastHCS                 Function to compute the FastHCS outlyingness index for 
			high-dimensional data-sets.
compPcaParams                 Internal function used to compute the FastHCS PCA model parameters.
DnaAlteration		Cytosine methylation beta values for a sample of 198 
			non-pathological human tissue specimens.
Ionosphere		Johns Hopkins University Ionosphere database.
MultipleFeatures         Fourier coefficients describing the shape of many hand written
			 replications of the numerals '0' and '1'.
numStarts                 Internal function used to compute number of starting subsets for the FastHCS algorithm.
plot.FastHCS		PCA diagnostic plot for object of class FastHCS.
quanf                 Internal function used to compute the size of the h-subsets used in FastHCS based on the input parameter alpha.
signFlip                 Performs the sign flip operation on a matrix of loadings.
Tablets                 Near-infrared (NIR) spectroscopy of a sample of 310 tablets.
}
}
\author{
Kaveh Vakili (primary programmer), Eric Schmitt
Maintainer: Kaveh Vakili <vakili.kaveh.email@gmail.com>
}
\keyword{package}
