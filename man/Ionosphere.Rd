\name{Ionosphere}
\alias{Ionosphere}
\title{Johns Hopkins University Ionosphere data set.}
\description{
   This radar data was collected by a system in Goose Bay, Labrador.  This
   system consists of a phased array of 16 high-frequency antennas with a
   total transmitted power on the order of 6.4 kilowatts.  See the paper
   for more details.  The targets were free electrons in the ionosphere.
   "Good" radar returns are those showing evidence of some type of structure 
   in the ionosphere.  "Bad" returns are those that do not; their signals pass
   through the ionosphere.  

   Received signals were processed using an autocorrelation function whose
   arguments are the time of a pulse and the pulse number.  There were 17
   pulse numbers for the Goose Bay system.  Instances in this databse are
   described by 2 attributes per pulse number, corresponding to the complex
   values returned by the function resulting from the complex electromagnetic
   signal.

In general, the Ionosphere data set describes a binary classification task
 where two types of electrons are targeted in the ionosphere by the radar 
signals, those that show some structure (good) and those that do not (bad).

	In this application, we joined the all the 225 observations corresponding 
	to the so-called 'Good' instances to 126 observations corresponding to so-called 
	'bad' instances. The first 34 four columns are the data and the last one are the 
	labels (1 for 'Good' and 0 for 'bad'). In effect, we cast the 'bad' instances as 
	(hidden) outliers and the task of the outlier detection method will be to reveal 
	them.
}
\usage{
Ionosphere
}
\format{
  \describe{
	\item{Labels}{Observations with label "0" correspond to the so-called good instances and observations 
	with label "1" to so-called 'bad' ones.}
	\item{Column 2--32}{Received signals were processed using an autocorrelation function whose
   arguments are the time of a pulse and the pulse number.  There were 15
   pulse numbers for the Goose Bay system.  Instances in this databse are
   described by 2 attributes per pulse number, corresponding to the complex
   values returned by the function resulting from the complex electromagnetic
   signal.}
  }
}
\source{
 Sigillito, V. G., Wing, S. P., Hutton, L. V., \& Baker, K. B. (1989).
      Classification of radar returns from the ionosphere using neural 
      networks. Johns Hopkins APL Technical Digest, 10, 262-266.

Mostafa, S. A.; Adeola, A. M.; Yousif. Y. K.; Naser, N. A. Ionosphere classification.
 http://mitm613.blogspot.be/p/inos-fave-classification.html.
}
\examples{
data(Ionosphere)
alpha<-0.5
Q<-15
p<-ncol(Ionosphere[,-1])
ns<-numStarts(q=Q,eps=(1-alpha)*4/5)
RunExample<-FALSE
if(RunExample){
	Fit<-FastHCS(x=Ionosphere[,-1],q=Q,nSamp=ns,seed=1)
	colvec<-rep("orange",nrow(ionosphere))
	colvec[Ionosphere[,1]==1]<-"blue"
	plot(Fit,col=colvec,pch=16)

	badOut<-which(Fit$best)
	par(mfrow=c(2,1))
	matplot(Ionosphere[,-1][badOut,],type="l",lty=1,col="orange",ylab="Bad instances")
	matplot(Ionosphere[,-1][-badOut,],type="l",lty=1,col="blue",ylab="Good instances")

}
}
\keyword{datasets}
