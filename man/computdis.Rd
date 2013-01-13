\name{computdis}
\alias{computdis}
\title{Computes the rank k S.D and O.D of a data matrix with respect ot a subset}
\usage{
  computdis(x,best,k)
}
\arguments{
  \item{x}{data matrix X.}
  \item{best}{Subset of observations used to compute the 
svd decomposition of X.}
  \item{k}{Desired rank of the svd decomposition.}
}
\value{
   A  list with the following
  components:
	  \item{center}{The multivariate mean of the observations with indexes in best.}
	  \item{loadings}{The (rank k) loadings matrix of the observations with indexes in best.}
	  \item{eigenvalues}{The (k) eigenvalues of the observations with indexes in best.}
	\item{scores}{The scores (the projection of the centered data on the subspace spanned 
by the loadings matrix).}
	\item{reod}{The orthogonal distances of the centered data wrt to the subspace spanned 
	by the loadings matrix.}
	\item{resd}{The statistical distances of the data projected on the subspace spanned by the loadings matrix.}
	\item{k}{corresponds to min(rank(X),k).}
}
\description{
  This is an internal function not intended to be called 
  by the user.
}
\examples{
n<-100
p<-50
x<-matrix(rnorm(n*p),nc=p)
best<-1:60
k<-5
rdis<-computdis(x=x,best=best,k=k)
}
\author{
Kaveh Vakili
}
\keyword{multivariate}
\keyword{robust}

