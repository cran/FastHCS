\name{Wine}
\alias{Wine}
\title{Chemical and sensory analysis of red and white variants of the Portuguese "Vinho Verde" wine.}
\description{
  The full data set contains 1359 unique red wine and 3961 unique white wine samples. In this subset, we combine all 1359 
red wine samples with the first 1000 white wine samples. 
}
\usage{
Wine
}
\format{
  \describe{
	\item{Labels}{The observations with label '1' correspond to the red wine samples and the '0' to the white ones.}
	\item{Column 2--13}{variables: volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide,
	total sulfur dioxide, density,	pH, sulphates, alcohol and a quality score that takes values between one and ten.}
  }
}
\source{
Cortez, P. Cerdeira, A.  Almeida, F.  Matos,  T.  and Reis, J. (2009).
Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, Elsevier, 47, 547--553.
}
\examples{
data(Wine)
alpha<-0.5
Q<-11
x0<-Wine[,-1]
p<-ncol(x0)
ns<-NumStarts(k=Q,eps=(1-alpha)*4/5)
RunExample<-FALSE
if(RunExample){
	Fit<-FastHCS(x=x0,k=Q,nsamp=ns,seed=1)
	colvec<-rep("orange",nrow(Wine))
	colvec[Wine[,1]==1]<-"blue"
	SDIND<-Fit$rew.fit$sd/Fit$rew.fit$cutoff.sd
	ODIND<-Fit$rew.fit$od/Fit$rew.fit$cutoff.od
	plot(SDIND,ODIND,col=colvec,pch=16)
	abline(h=1,col="red",lty=2)
	abline(v=1,col="red",lty=2)
}
}
\keyword{datasets}
