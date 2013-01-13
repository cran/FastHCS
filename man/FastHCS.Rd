\name{FastHCS}
\alias{FastHCS}
\title{Computes the FastHCS high-dimensional outlyingness index.}
\usage{
  FastHCS(x,nsamp=NULL,alpha=0.5,k=10,seed=NULL)
}
\arguments{
  \item{x}{A numeric n (n>5*p) by p (p>1) matrix or data frame.}
   \item{nsamp}{a positive integer giving the number of resamples required;
          \code{"nsamp"} may not be reached if too many of the p-subsamples,
          chosen out of the observed vectors, are in a hyperplane.  If \code{"nsamp"} is
          omitted, it is calculated to provide a breakdown point of
          \code{"alpha"} with probability 0.99.}
 \item{alpha}{numeric parameter controlling the size of the active subsets,
	 i.e., \code{"h=quanf(alpha,n,p)"}.  Allowed
          values are between 0.5 and 1 and the default is 0.5.}
\item{k}{number of principal components to compute. Note that p>k>1. Default is k=10.}
\item{seed}{starting value for random generator. Default is seed = NULL}
}
\value{
 A  list with components:
  \item{outi}{The n-vector of rank reduced projection distances. Typically for internal use.}
  \item{obj}{The FastHCS objective function corresponding to H*, the most congruent low rank h-subset.}
  \item{raw.fit}{A list with components:}
	 \enumerate{
		\item best: the indexes of the h members of the H*, the raw FastHCS optimal subset.
		\item od: the orthogonal distances of the centered data wrt to the subspace spanned 
		by the loadings matrix. This is the raw FastHCS outlyingness index.
		\item sd: the score distances of the data projected on the subspace spanned by the loadings matrix.
		the most congruent rank reduced subset.
		\item scores: the scores (the projection of the centered data on the subspace spanned 
		by the loadings matrix).
		\item eigenvalues: the (k) eigenvalues of the observations with indexes in best.
		\item loadings: the (rank k) loadings matrix of the observations with indexes in best.
		\item center: the p-vector of column means of the observations with indexes in best. 
		\item cutoff.od: the cutoff for the vector of orthogonal distances.
		\item cutoff.sd: the cutoff for the vector of score distances.
	  }
  \item{rew.fit}{A list with components:}
	 \enumerate{
		\item best: the indexes of the h members of the H*_+, the reweighted FastHSC subset. 
		\item od: the orthogonal distances of the centered data wrt to the subspace spanned 
		by the loadings matrix. This is the re-weighted ('final') FastHCS outlyingness index.
		\item sd: the score distances of the data projected on the subspace spanned by the loadings matrix.
		the most congruent rank reduced subset.
		\item scores: the scores (the projection of the centered data on the subspace spanned 
		by the loadings matrix).
		\item eigenvalues: the (k) eigenvalues of the observations with indexes in best.
		\item loadings: the (rank k) loadings matrix of the observations with indexes in best.
		\item center: the p-vector of column means of the observations with indexes in best. 
		\item cutoff.od: the cutoff for the vector of orthogonal distances.
		\item cutoff.sd: the cutoff for the vector of score distances.
	  }
}
\description{
   Computes a fast and robust outlyingness index for the rank k approximation to 
	a n by p matrix of multivariate data.
}
\examples{
## generate data
set.seed(123)
n<-100
p<-30
q<-5
x0<-matrix(rnorm(n*p),nc=p)
x0[1:30,]<-matrix(rnorm(30*p,4.5,1/100),nc=p)
z<-c(rep(0,30),rep(1,70))
nstart<-NumStarts(k=q,eps=0.4)
results<-FastHCS(x=x0,nsamp=nstart)
z[results$rew.fit$best]

#testing exact fit
set.seed(123)
n<-100
p<-5
q<-4
x0<-matrix(rnorm(n*p),nc=p)
x0[1:30,]<-matrix(rnorm(30*p,4.5,1/100),nc=p)
x0[31:100,4:5]<-x0[31:100,2]
z<-c(rep(0,30),rep(1,70))
nstart<-NumStarts(k=q,eps=0.4)
results<-FastHCS(x=x0,nsamp=nstart,k=q)
z[results$rew.fit$best]
results$obj

#testing rotation equivariance
set.seed(123)
n<-100
p<-10
q<-3
x0<-scale(matrix(rnorm(n*p),nc=p))
A<-diag(rep(1,p))
A[1:2,1:2]<-c(0,1,-1,0)
x1<-x0%*%A
nstart<-NumStarts(k=q,eps=0.4)
r0<-FastHCS(x=x0,nsamp=nstart,k=q,seed=1)
r1<-FastHCS(x=x1,nsamp=nstart,k=q,seed=1)
max(abs(log(r1$rew.fit$eigenvalues[1:q]/r0$rew.fit$eigenvalues[1:q])))
}
\references{
	Vakili, K. and Schmitt, E. (2014).
	Finding High-Dimensional Outliers With FastHCS.
	(http://arxiv.org/abs/1402.3514)
}
\author{
	Kaveh Vakili
}
\keyword{multivariate}
\keyword{robust}

