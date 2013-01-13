#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <fstream>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <vector>
#include <random>
#include <R.h>
#include <Rmath.h>

#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/SVD>

using namespace std;
using namespace Eigen;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::VectorXi;
using Eigen::RowVectorXf;
std::mt19937 mt;

struct IdLess {					//internal function.
    template <typename T>
    IdLess(T iter) : values(&*iter) {}
    bool operator()(int left,int right){
        return values[left]<values[right];
    }
    float const* values;
};
float GetUniform(){
    static std::uniform_real_distribution<float> Dist(0,1);
    return Dist(mt);
}
void GetSmallest(const VectorXf& r,int h,const MatrixXf& x,MatrixXf& xSub,VectorXi& RIndex){
	const int n=x.rows();
	VectorXi SIndx2(n);
	SIndx2.setLinSpaced(n,0,n-1);
	std::nth_element(SIndx2.data(),SIndx2.data()+h,SIndx2.data()+SIndx2.size(),IdLess(r.data()));
	for (int i=0;i<h;i++) 	xSub.row(i)=x.row(SIndx2(i));
	RIndex.head(h)=SIndx2.head(h);	
}
VectorXi SampleR(const int m,const int p){
	int i,j,nn=m;
	VectorXi ind(nn);
	ind.setLinSpaced(nn,0,nn-1);
	VectorXi y(p);
    	for(i=0;i<p;i++){
		j=GetUniform()*nn;
		y(i)=ind(j);
		ind(j)=ind(--nn);
    	}
	return y;		
}
VectorXf FindLine(const MatrixXf& xSub,const int& h){
	const int p=xSub.cols();
	VectorXi QIndexp=SampleR(h,p);
	MatrixXf A(p,p);
	for(int i=0;i<p;i++)	A.row(i)=xSub.row(QIndexp(i));
	VectorXf bt=VectorXf::Ones(p);
	return(A.lu().solve(bt));
}
VectorXf OneProj(const MatrixXf& x,const MatrixXf& xSub,const int& h,const VectorXi& RIndex,const int h_m){
	const int p=x.cols(),n=x.rows();
	VectorXf beta(p);
	beta=FindLine(xSub,h);
	VectorXf praj(n);
	praj=((x*beta).array()-1.0f).array().abs2();
	praj/=beta.squaredNorm();
	VectorXf prej(h);
	for(int i=0;i<h;i++)	prej(i)=praj(RIndex(i));
	float prem=prej.head(h).mean(),tol=1e-8;
	if(prem<tol){	
		const int n=praj.size();
		VectorXf d_resd=VectorXf::Zero(n);
		d_resd=(praj.array()<tol).select(1.0f,d_resd);
		if((d_resd.sum())>=h_m){
			prem=1.0f;
		} else {
			float maxin=praj.maxCoeff();
			d_resd=(praj.array()<tol).select(maxin,praj);
			prem=d_resd.minCoeff();
		}
	}
	return praj/=prem;
}
float SubsetRankFun(const MatrixXf& x,const MatrixXf& xSub,const int& h,const VectorXi& RIndex){
	const int p=x.cols(),n=x.rows();
	VectorXf beta(p);
	beta=FindLine(xSub,h);
	VectorXf praj(n);
	praj=((x*beta).array()-1.0f).array().abs2();
	praj/=beta.squaredNorm();
	VectorXf proj=praj;
	float fin=1.0f,prem;
	std::nth_element(proj.data(),proj.data()+h,proj.data()+proj.size());	
	VectorXf prej(h);
	for(int i=0;i<h;i++)	prej(i)=praj(RIndex(i));
	prem=proj.head(h).mean();
	if(prem>1e-8)	fin=prej.head(h).mean()/prem;
	return fin;
}
float Main(MatrixXf& x,const int& k0,const int& J,const int& k1,const int& K,VectorXf& dP,const int& h_m,VectorXi& samset){
	int p=x.cols(),n=x.rows(),ni=samset.size();
	int h=K+1,i=0;
	VectorXi RIndex(h_m);
	RIndex.head(h)=SampleR(ni,h);
	for(i=0;i<h;i++) RIndex(i)=samset(RIndex(i));	
	MatrixXf xSub(h_m,p);
	for(i=0;i<h;i++) xSub.row(i)=x.row(RIndex(i));	
	RowVectorXf xSub_mean(p);
	xSub_mean=xSub.topRows(h).colwise().mean();	
	x.rowwise()-=xSub_mean;					
	xSub.rowwise()-=xSub_mean;
	JacobiSVD<MatrixXf> svd(xSub.topRows(h),ComputeThinV);
	MatrixXf w(n,K); 
	w=(x*svd.matrixV().topLeftCorner(p,K));
	MatrixXf wSub(h_m,K);
	for(i=0;i<h;i++) wSub.row(i)=w.row(RIndex(i));	
	VectorXi hl;
	hl.setLinSpaced(J+1,h,h_m);
	h=hl(0);	
	for(int j=0;j<J;j++){			//growing step
		dP=VectorXf::Zero(n);
		for(int i=0;i<k0;i++) dP+=OneProj(w,wSub,h,RIndex,h_m);
		h=hl(j+1);
		GetSmallest(dP,h,w,wSub,RIndex);
	}
	VectorXf fin(k1);
	for(int i=0;i<k1;i++) fin(i)=SubsetRankFun(w,wSub,h,RIndex);
	return fin.array().log().mean();
}
void CStep(VectorXi& RIdx,MatrixXf& x,const int K,const int m){
	const int n=x.rows(),p=x.cols(),h=RIdx.size();
	int w2=1,w3=1;
	MatrixXf xSb(h,p);
	for(int i=0;i<m;i++) 	xSb.row(i)=x.row(RIdx(i));
	RowVectorXf xSb_mean(p);
	xSb_mean=xSb.topRows(m).colwise().mean();	
	xSb.rowwise()-=xSb_mean;
	x.rowwise()-=xSb_mean;
	JacobiSVD<MatrixXf> svd(xSb.topRows(m),ComputeThinV);
	xSb_mean.head(K)=svd.singularValues().head(K);
	float w0,w1=xSb_mean(K-1);
	VectorXf v_sd(n); 
	MatrixXf w4(n,K); 
	if(w1>1e-8){
		w1=log(std::numeric_limits<float>::max());
		w4=(x*svd.matrixV().topLeftCorner(p,K));
		for(int i=0;i<K;i++)	w4.col(i)/=xSb_mean(i); 
		v_sd=w4.cwiseAbs2().rowwise().sum();
	} else {
		w2=0;
		w3=0;
	}
	VectorXi dIn(n);
	while(w2){	
		dIn.setLinSpaced(n,0,n-1);
		std::nth_element(dIn.data(),dIn.data()+h,dIn.data()+dIn.size(),IdLess(v_sd.data()));
		for(int i=0;i<h;i++) 	xSb.row(i)=x.row(dIn(i));
		xSb_mean=xSb.colwise().mean();	
		xSb.rowwise()-=xSb_mean;
		x.rowwise()-=xSb_mean;
		JacobiSVD<MatrixXf> svd(xSb,ComputeThinV);
		xSb_mean.head(K)=svd.singularValues().head(K);
		if(xSb_mean(K-1)>1e-8){
			w0=w1;
			w1=log(xSb_mean(K-1));
			w4=(x*svd.matrixV().topLeftCorner(p,K));
			for(int i=0;i<K;i++)	w4.col(i)/=xSb_mean(i); 
			v_sd=w4.cwiseAbs2().rowwise().sum();
			(w0-w1<1e-3)?(w2=0):(w2=1);
		} else {
			w2=0;
		}
	}
	if(w3)	RIdx=dIn.head(h);
} 
int compudist(MatrixXf& x,const int h_m,const int K,VectorXf& v_od,VectorXi& RIndex){
	const int p=x.cols(),n=x.rows();
	MatrixXf xSb(h_m,p);
	GetSmallest(v_od,h_m,x,xSb,RIndex);
	RowVectorXf xSb_mean(p);
	xSb_mean=xSb.colwise().mean();
	x.rowwise()-=xSb_mean;					
	xSb.rowwise()-=xSb_mean;
	JacobiSVD<MatrixXf> svd(xSb,ComputeThinV);
	int j=svd.nonzeroSingularValues()-1;
	const int Q=min(K,j);
	MatrixXf wOut(n,Q);
	wOut=(x*svd.matrixV().topLeftCorner(p,Q));
	v_od=((x-wOut*svd.matrixV().topLeftCorner(p,Q).adjoint()).cwiseAbs2().rowwise().sum()).array().sqrt(); 
	return(Q);
}
float cut_off_od(Ref<VectorXf> v_sd,const int n){
	const int h_m=v_sd.size();
	float temp0=v_sd.sum();
	temp0/=(float)h_m;		//mean
	v_sd.array()-=temp0;
	float temp1=v_sd.array().abs2().sum();
	temp1/=(float)(h_m-1);
	temp1=sqrt(temp1);		//sd
	float temp2=(float)sqrt(qchisq((h_m/(double)(n+(h_m==n))),1.0,1,0));
	temp1/=temp2;			//srob
	temp2=(float)qnorm(0.975,(double)temp0,(double)temp1,1,0);
	return(pow(temp2,3.0f/2.0f));	//cut_off
}
void CStepPrep(const MatrixXf& x,VectorXf& v_od,VectorXi& RIdx,const int K,const int h_m){
	const int n=RIdx.size();
	float temp0=2.0f/3.0f;
	int j=0,m=0;
	VectorXf v_sd(n);
	for(int i=0;i<h_m;i++)	v_sd(i)=v_od(RIdx(i));
	v_sd.head(h_m)=v_sd.head(h_m).array().pow(temp0); 	
	temp0=cut_off_od(v_sd.head(h_m),n);
	VectorXi LIdx(n);
	VectorXi CIdx(n);			//bests.
	CIdx.setZero(n);
	for(int i=0;i<h_m;i++)	CIdx(RIdx(i))=1;	
	VectorXi VIdx(h_m);
	for(int i=0;i<n;i++){
		if(v_od(i)<=temp0){
			if(CIdx(i))	VIdx(m++)=j;
			LIdx(j++)=i;
		}		
	}
	if(j>h_m){
		const int p=x.cols();
		MatrixXf xSb(j,p);
		for(int i=0;i<j;i++)	xSb.row(i)=x.row(LIdx(i));
		CStep(VIdx,xSb,K,m);
		for(int i=0;i<h_m;i++)	RIdx(i)=LIdx(VIdx(i));
	}
}	
int checkout(MatrixXf& x,VectorXi& VIdx,VectorXf& v_od,VectorXf& v_sd,const int h_m,VectorXf& cutoffs,const int K){
	const int n=x.rows(),p=x.cols();
	float temp0=2.0f/3.0f;
	MatrixXf xSb(h_m,p);
	for(int i=0;i<h_m;i++)	xSb.row(i)=x.row(VIdx(i));
	RowVectorXf xSb_mean(p);
	xSb_mean=xSb.colwise().mean();
	x.rowwise()-=xSb_mean;					
	xSb.rowwise()-=xSb_mean;
	JacobiSVD<MatrixXf> svd(xSb,ComputeThinV);	
	MatrixXf w(n,K);
	w=(x*svd.matrixV().topLeftCorner(p,K));
	v_od=((x-w*svd.matrixV().topLeftCorner(p,K).adjoint()).cwiseAbs2().rowwise().sum()).array().sqrt(); 
	for(int i=0;i<h_m;i++)	v_sd(i)=v_od(VIdx(i));
	v_sd.head(h_m)=v_sd.head(h_m).array().pow(temp0); 	
	temp0=cut_off_od(v_sd.head(h_m),n);			//cut-off_od
	xSb_mean.head(K)=svd.singularValues().head(K);
	float temp1=1.0f/sqrt((float)(h_m-1));
	xSb_mean.head(K).array()*=temp1;
	for(int i=0;i<K;i++)  w.col(i).array()/=xSb_mean(i);
	v_sd=(w.cwiseAbs2().rowwise().sum()).array().sqrt(); 
	float temp2=(float)sqrt(qchisq(0.975,(double)K,1,0)/qchisq(h_m/(double)(n+(h_m==n)),(double)K,1,0));
	VIdx.setLinSpaced(n,0,n-1);
	std::nth_element(VIdx.data(),VIdx.data()+h_m-1,VIdx.data()+VIdx.size(),IdLess(v_sd.data()));
	temp2*=v_sd(VIdx(h_m-1));	
	int j=0;
	for(int i=0;i<n;i++){
		if(v_od(i)<=temp0 && v_sd(i)<=temp2) VIdx(j++)=i+1;
	}
	cutoffs<<temp0,temp2;
	return(j);
}
extern "C"{
	void FastHCS(int* rn,		//1
			int* p,		//2
			int* k0,	//3
			float* xi,	//4
			int* k1,	//5
			int* k2,	//6
			int* nsmp,	//7
			int* J,		//8
			float* objfunC,	//9
			int* seed,	//10
			int* ck,	//11
			int* rni,	//12
			int* rraw,	//13
			int* rrew,	//14
			int* rS,	//15
			float* rod,	//16
			float* rsd,	//17
			float* rc,	//18
			int* rh,	//19
			float* pco	//20
		){
		const int n=*rn,ni=*rni,ik0=*k0,iJ=*J,ik1=*k1,K=*k2,ih_m=*rh;
		float objfunA,objfunB=*objfunC;
		mt.seed(*seed);
		MatrixXf x=Map<MatrixXf>(xi,n,*p);
		VectorXi icK=Map<VectorXi>(ck,ni);	
		VectorXf DpA=VectorXf::Zero(n);
		VectorXf DpB=VectorXf::Zero(n);
		for(int i=0;i<*nsmp;i++){			//for i=0 to i<#p-subsets.
			objfunA=Main(x,ik0,iJ,ik1,K,DpA,ih_m,icK);
			if(objfunA<objfunB){
				objfunB=objfunA;
				DpB=DpA;
			}
		}
		*objfunC=objfunB;
		VectorXi RIdx(n);
		VectorXf coff(2);
		Map<VectorXf>(pco,n)=DpB;		
		int Q=compudist(x,ih_m,K,DpB,RIdx);	//changes dPB,x. 
		Map<VectorXi>(rraw,ih_m)=RIdx.head(ih_m);
		CStepPrep(x,DpB,RIdx,Q,ih_m);		//uses dPB
		int S=checkout(x,RIdx,DpA,DpB,ih_m,coff,Q);
		Map<VectorXi>(rrew,S)=RIdx.head(S);
		Map<VectorXf>(rod,n)=DpA;
		Map<VectorXf>(rsd,n)=DpB;
		Map<VectorXf>(rc,2)=coff;
		*k2=Q;
		*rS=S;
		
	}
}
