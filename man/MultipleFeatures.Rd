\name{MultipleFeatures}
\alias{MultipleFeatures}
\title{Fourier coefficients describing the shape of many hand written replications of the numerals '0' and '1'.}
\description{
  A data frame with the subset of the 'Multiple Features'
 dataset corresponding to the sample of '0' and '1' numerals. 
}
\usage{
MultipleFeatures
}
\format{
  \describe{
	\item{Labels}{Numerals.}
	\item{Column 2--77}{Fourier coefficients describing the shape of each observation.}
  }
}
\source{
Van Breukelen, M. Duin, R.P.W. Tax, D.M.J. and  Den Hartog, J.E. (1998).
Handwritten digit recognition by combined classifiers, Kybernetika, vol. 34, 381--386.
}
\examples{
data(MultipleFeatures)
alpha<-0.5
Q<-15
x0<-MultipleFeatures[,-1]
p<-ncol(x0)
ns<-NumStarts(k=Q,eps=(1-alpha)*4/5)
RunExample<-FALSE
if(RunExample){
	Fit<-FastHCS(x=x0,k=Q,nsamp=ns,seed=1)
	colvec<-rep("orange",nrow(MultipleFeatures))
	colvec[MultipleFeatures[,1]==1]<-"blue"
	SDIND<-Fit$rew.fit$sd/Fit$rew.fit$cutoff.sd
	ODIND<-Fit$rew.fit$od/Fit$rew.fit$cutoff.od
	plot(SDIND,ODIND,col=colvec,pch=16)
	abline(h=1,col="red",lty=2)
	abline(v=1,col="red",lty=2)
}
}
\keyword{datasets}
