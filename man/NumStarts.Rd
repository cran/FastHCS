\name{NumStarts}
\alias{NumStarts}
\title{Computes the number of starting q subset}
\usage{
  NumStarts(k,gamma=0.99,eps=0.5)
}
\arguments{
  \item{k}{number of desiered components.}

  \item{gamma}{desired probability of having at least one 
	clean starting p-subset.}

  \item{eps}{suspected contamination rate of the sample.}

}
\value{
  An integer number of starting q-subset.
}
\description{
  This is an internal function not intended to be called 
  by the user.
}
\examples{
NumStarts(k=3,gamma=0.99,eps=0.4)
}
\author{
Kaveh Vakili
}
\keyword{multivariate}
\keyword{robust}
