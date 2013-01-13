numStarts<-function(q,gamma=0.99,eps=0.5){
	if(q>25)	stop("q too large.")
	if(gamma>=1)	stop("gamma should be smaller than 1.")
	if(gamma<=0)	stop("gamma should be larger than 0.")
	if(eps>0.5)	stop("eps should be smaller than 1/2.")
	if(eps<=0)	stop("eps should be larger than 0.")	
	ns0<-ceiling(log(1-gamma)/log(1-(1-(eps))^(q+1)))
	ord<-10^floor(log10(ns0))
	ceiling(ns0/ord)*ord
}
FastHCS<-function(x,nSamp=NULL,alpha=0.5,q=10,seed=1){#x<-x0;nSamp<-100;alpha<-0.5;q=10;seed=NULL
	q0<-q1<-25;J<-5;qo<-q
	m1<-"seed should be an integer in [0,2**31]."
	if(!is.null(seed)){
		if(!is.finite(seed))		stop(m1)
		if(!is.numeric(seed))		stop(m1)
		if(seed<0)			stop(m1)
		if(is.na(as.integer(seed)))	stop(m1)
	}
	seed<-as.integer(seed)+2
	x<-data.matrix(x)
	na.x<-complete.cases(x)
	if(!is.numeric(alpha))		stop("alpha should be numeric")
	if(alpha<0.5 | alpha>=1)	stop("alpha should be in (0.5,1(.")
	if(sum(na.x)!=nrow(x))		stop("Your data contains NA.")
	n<-nrow(x)
	p<-ncol(x)
	if(q>=p)		stop("q should satisfy q<p.")
	if(p<2|q<2)		stop("Univariate FastHCS is not implemented.")
	if(q>25)		stop("FastHCS only works for dimensions q<=25.")
	if(is.null(nSamp)) 	nSamp<-numStarts(q,eps=(1-alpha)) 
	h<-quanf(n=n,p=q,alpha=0.5)
	Dp<-rep(1.00,n);
	q0<-max(q0,q);
	q1<-max(q1,q);
	objfunC<-1e3;
	icandid<-1:n-1
	ni<-length(icandid)
	rraw<-rep(0,h)
	rrew<-rep(0,n)
	sd.d<-od.d<-rep(0,n)
	co<-rep(0,2);
	fitd<-.C("FastHCS",	
		as.integer(nrow(x)),	#1
		as.integer(ncol(x)),	#2
		as.integer(q0),		#3
		as.single(x),		#4
		as.integer(q1),		#5
		as.integer(q),		#6
		as.integer(nSamp),	#7
		as.integer(J),		#8
		as.single(objfunC),	#9
		as.integer(seed),	#10
		as.integer(icandid),	#11
		as.integer(ni),		#12
		as.integer(rraw),	#13
		as.integer(rrew),	#14
		as.integer(0),		#15
		as.single(od.d),	#16
		as.single(sd.d),	#17
		as.single(co),		#18
		as.integer(h),		#19
		as.single(sd.d),	#20
	PACKAGE="FastHCS")
	eStep<-compPcaParams(x=x,best=fitd[[14]][1:(fitd[[15]])],q=fitd[[6]]);#efficiency Step
	A1<-list(rawBest = fitd[[13]],obj=as.numeric(fitd[[9]]), rawDist=fitd[[20]], best=fitd[[14]][1:(fitd[[15]])],loadings=eStep$loadings,eigenvalues=eStep$eigenvalues,center=eStep$center,od=fitd[[16]],sd=fitd[[17]],cutoff.od=fitd[[18]][1],cutoff.sd=fitd[[18]][2])
	class(A1)<-"FastHCS"
	return(A1)
}
quanf<-function(n,p,alpha)	return(floor(2*floor((n+p+1)/2)-n+2*(n-floor((n+p+1)/2))*alpha))
signFlip<-function(loadings) 	apply(loadings,2,function(x) if (x[which.max(abs(x))]<0) -x else x)
compPcaParams<-function(x,best,q){
	h<-length(best)
	n<-nrow(x)
	center<-colMeans(x[best,])
	centeredx<-sweep(x,2,center,FUN="-")
	eSVD<-svd(centeredx[best,]/sqrt(h-1),nu=0); # efficiency step SVD
	if(eSVD$d[q]<1e-6) q<-sum(eSVD$d>1e-6)-1
	eval<-as.vector(eSVD$d)
	loadings<-eSVD$v[,1:q,drop=FALSE];
	eigenvalues<-eval**2
	list(center=center,loadings=loadings,eigenvalues=eigenvalues)
}
plot.FastHCS<-function(x,col="black",pch=16,...){
	SDIND<-x$sd
	ODIND<-x$od
	plot(SDIND,ODIND,col=col,pch=pch,xlab="Robust score distance",ylab="Robust orthogonal distance",main="Robust PCA")
	abline(h=x$cutoff.od,col="red",lty=2)
	abline(v=x$cutoff.sd,col="red",lty=2)
}
