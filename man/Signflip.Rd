\name{Signflip}
\alias{Signflip}
\title{Carries the signflip adjustment of a loading matrix}
\usage{
  Signflip(loadings)
}
\arguments{
  \item{loadings}{A matrix of loadings.}

}
\value{
  A (eventually sign fliped) loading matrix.
}
\description{
THis function solves the sign indeterminancy of the loadings by 
setting the maximum element in a singular vector to be positive. 
}
\examples{
X<-diag(10)
X[1,1]<--2
W<-Signflip(X)
W[1,1]
}
\author{
Kaveh Vakili
}
\keyword{multivariate}
\keyword{robust}
