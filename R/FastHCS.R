NumStarts<-function(k,gamma=0.99,eps=0.5){
	if(k>25)	stop("k too large.")
	if(gamma>=1)	stop("gamma should be smaller than 1.")
	if(gamma<=0)	stop("gamma should be larger than 0.")
	if(eps>0.5)	stop("eps should be smaller than 1/2.")
	if(eps<=0)	stop("eps should be larger than 0.")	
	ns0<-ceiling(log(1-gamma)/log(1-(1-(eps))^(k+1)))
	ord<-10^floor(log10(ns0))
	ceiling(ns0/ord)*ord
}
FastHCS<-function(x,nsamp=NULL,alpha=0.5,k=10,seed=NULL){#x<-x0;nsamp<-100;alpha<-0.5;k=10;seed=NULL
	k0<-k1<-25;J<-5;ko<-k
	if(is.null(seed))	seed<-floor(runif(1,-2^31,2^31))
	seed<-as.integer(seed)+1
	x<-data.matrix(x)
	na.x<-complete.cases(x)
	if(!is.numeric(alpha))	stop("alpha should be numeric")
	if(alpha<0.5 | alpha>=1)	stop("alpha should be in (0.5,1(.")
	if(sum(na.x)!=nrow(x))  stop("Your data contains NA.")
	x1<-sweep(x,2,colMeans(x),FUN="-")
	vd<-svd(x1/sqrt(nrow(x)-1),nu=0)
	x2<-x1%*%Signflip(vd$v)
	n<-nrow(x2)
	if(nrow(unique(x2))<n)	stop("Your dataset contains duplicated rows. Please remove them.") 
	p<-ncol(x2)
	if(k>=p)		stop("k should satisfy k<p.")
	if(p<2|k<2)		stop("Univariate FastHCS is not implemented.")
	if(k>25)		stop("FastHCS only works for dimensions k<=25.")
	if(is.null(nsamp)) 	nsamp<-NumStarts(k,eps=(1-alpha)) 
	h<-quanf(n=n,p=k,alpha=alpha)
	Dp<-rep(1.00,n);
	k0<-max(k0,k);
	k1<-max(k1,k);
	objfunC<-1e3;
	icandid<-1:n-1
	ni<-length(icandid)
	fitd<-.C("FastHCS",as.integer(nrow(x2)),as.integer(ncol(x2)),as.integer(k0),as.single(x2),as.integer(k1),as.single(Dp),as.integer(k),as.integer(nsamp),as.integer(J),as.single(objfunC),as.integer(seed),as.integer(icandid),as.integer(ni),PACKAGE="FastHCS")
	outi<-as.numeric(fitd[[6]])
	if(is.nan(outi)[1])	stop("too many singular subsets encoutered! Try with lower values of k.")
	best<-which(outi<=median(outi))
	rdis<-computdis(x=x,best=best,k=k)
	rawF<-list(best=best,od=rdis$reod,sd=rdis$resd,scores=rdis$scores,eigenvalues=rdis$eigenvalues,loadings=rdis$loadings,center=rdis$center,oindx=outi)
	rdis<-computdis(x=x,best=best,k=k)
	rwds<-list(trob=mean(rdis$reod[best]^(2/3)),srob=1/sqrt(qchisq(h/n,1))*sd(rdis$reod[best]^(2/3)))
	cofo<-sqrt(qnorm(0.975,rwds$trob,rwds$srob)^3)
	best<-which(rdis$reod<=cofo)
	rdis<-computdis(x=x,best=best,k=k);
    	coso<-sqrt(qchisq(0.975,rdis$k))
	rewF<-list(best=best,od=rdis$reod,sd=rdis$resd,scores=rdis$scores,eigenvalues=rdis$eigenvalues,loadings=rdis$loadings,center=rdis$center,cutoff.od=cofo,cutoff.sd=coso)
	list(outi=outi,obj=as.numeric(fitd[[10]]),k=k,rew.fit=rewF,raw.fit=rawF)
}
computdis<-function(x,best,k){
	cent<-colMeans(x[best,])
	xcen<-sweep(x,2,cent,FUN="-")
	rsvd<-svd(xcen[best,]/sqrt(length(best)-1),nu=0);
	if(rsvd$d[k]<1e-6) k<-sum(rsvd$d>1e-6)
	eval<-as.vector(rsvd$d[1:k])
	lods<-rsvd$v[,1:k,drop=FALSE];
	scor<-xcen%*%lods
	resd<-sweep(scor,2,eval,FUN="/")
	resd<-sqrt(rowSums(resd*resd))
	reod<-xcen-xcen%*%lods%*%t(lods);
	reod<-sqrt(rowSums(reod*reod));
	list(center=cent,loadings=lods,eigenvalues=eval**2,scores=scor,reod=reod,resd=resd,k=k)
}
quanf<-function(n,p,alpha)	return(floor(2*floor((n+p+1)/2)-n+2*(n-floor((n+p+1)/2))*alpha))
Signflip<-function(loadings) apply(loadings,2,function(x) if (x[which.max(abs(x))]<0) -x else x)
