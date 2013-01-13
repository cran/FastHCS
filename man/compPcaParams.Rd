\name{compPcaParams}
\alias{compPcaParams}
\title{Computes the center vector, eigenvalues and loading matrix corresponding to a PCA model of a data matrix with respect to a subset of observations in a data set}
\usage{
  compPcaParams(x,best,q)
}
\arguments{
  \item{x}{A data matrix x.}
  \item{best}{Subset of observations in x for on which the SVD decomposition will be computed.}
  \item{q}{Desired rank of the SVD decomposition.}
}
\value{
   A  list with the following components:
	  \item{center}{The multivariate mean of the observations with indexes in best.}
	  \item{loadings}{The (rank q) loadings matrix of the observations with indexes in best.}
	  \item{eigenvalues}{The eigenvalues of the observations with indexes in best.}
}
\description{
  This function is used in FastHCS to compute the parameter estimates of the PCA models used at different steps of the algorithm.
  It is an internal function not intended to be called 
  by the user. 
}
\examples{
n<-100
p<-50
x<-matrix(rnorm(n*p),nc=p)
best<-1:60
q<-5
pcaParams<-compPcaParams(x=x,best=best,q=q)
}
\author{
Kaveh Vakili, Eric Schmitt
}
\keyword{multivariate}
\keyword{robust}

