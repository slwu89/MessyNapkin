#lnam - Fit a linear network autocorrelation model
#y = r1 * W1 %*% y + X %*% b + e, e = r2 * W2 %*% e + nu
#y =  (I-r1*W1)^-1%*%(X %*% b + e)
#y = (I-r1 W1)^-1 (X %*% b + (I-r2 W2)^-1 nu)
#e = (I-r2 W2)^-1 nu
#e = (I-r1 W1) y - X b
#nu = (I - r2 W2) [ (I-r1 W1) y - X b ]
#nu = (I-r2 W2) e
lnam<-function(y,x=NULL,W1=NULL,W2=NULL,theta.seed=NULL,null.model=c("meanstd","mean","std","none"),method="BFGS",control=list(),tol=1e-10){
   #Define the log-likelihood functions for each case
   agg<-function(a,w){
     m<-length(w)
     n<-dim(a)[2]
     mat<-as.double(matrix(0,n,n))
     matrix(.C("aggarray3d_R",as.double(a),as.double(w),mat=mat,as.integer(m), as.integer(n),PACKAGE="sna",NAOK=TRUE)$mat,n,n)
   }
  #Estimate covariate effects, conditional on autocorrelation parameters
  betahat<-function(y,X,W1a,W2a){
    if(nw1==0){
      if(nw2==0){
        return(qr.solve(t(X)%*%X,t(X)%*%y))
      }else{
        tXtW2aW2a<-t(X)%*%t(W2a)%*%W2a
        return(qr.solve(tXtW2aW2a%*%X,tXtW2aW2a%*%y))
      }
    }else{
      if(nw2==0){
        return(qr.solve(t(X)%*%X,t(X)%*%W1a%*%y))
      }else{
        tXtW2aW2a<-t(X)%*%t(W2a)%*%W2a
        qr.solve(tXtW2aW2a%*%X,tXtW2aW2a%*%W1a%*%y)
      }
    }
  }
  #Estimate predicted means, conditional on other effects
  muhat<-function(y,X,W1a,W2a,betahat){
    if(nx>0)
      Xb<-X%*%betahat
    else
      Xb<-0
    switch((nw1>0)+2*(nw2>0)+1,
      y-Xb,
      W1a%*%y-Xb,
      W2a%*%(y-Xb),
      W2a%*%(W1a%*%y-Xb)
    )
  }
  #Estimate innovation variance, conditional on other effects
  sigmasqhat<-function(muhat){
    t(muhat)%*%muhat/length(muhat)
  }
  #Model deviance (for use with fitting rho | beta, sigma)
  n2ll.rho<-function(rho,beta,sigmasq){
    #Prepare ll elements according to which parameters are present
    if(nw1>0){
      W1a<-diag(n)-agg(W1,rho[1:nw1])
      W1ay<-W1a%*%y
      adetW1a<-abs(det(W1a))
    }else{
      W1ay<-y
      adetW1a<-1
    }
    if(nw2>0){
      W2a<-diag(n)-agg(W2,rho[(nw1+1):(nw1+nw2)])
      tpW2a<-t(W2a)%*%W2a
      adetW2a<-abs(det(W2a))
    }else{
      tpW2a<-diag(n)
      adetW2a<-1
    }
    if(nx>0){
      Xb<-x%*%beta
    }else{
      Xb<-0
    }
    #Compute and return
    n*(log(2*pi)+log(sigmasq)) + t(W1ay-Xb)%*%tpW2a%*%(W1ay-Xb)/sigmasq - 2*(log(adetW1a)+log(adetW2a))
  }
  #Model deviance (general purpose)
  n2ll<-function(W1a,W2a,sigmasqhat){
    switch((nw1>0)+2*(nw2>0)+1,
      n*(1+log(2*pi)+log(sigmasqhat)),
      n*(1+log(2*pi)+log(sigmasqhat))-2*log(abs(det(W1a))),
      n*(1+log(2*pi)+log(sigmasqhat))-2*log(abs(det(W2a))),
      n*(1+log(2*pi)+log(sigmasqhat))- 2*(log(abs(det(W1a)))+log(abs(det(W2a))))
    )
  }
  #Conduct a single iterative refinement of a set of initial parameter estimates
  estimate<-function(parm,final=FALSE){
    #Either aggregate the weight matrices, or NULL them
    if(nw1>0)
      W1a<-diag(n)-agg(W1,parm$rho1)
    else
      W1a<-NULL
    if(nw2>0)
      W2a<-diag(n)-agg(W2,parm$rho2)
    else
      W2a<-NULL
    #If covariates were given, estimate beta | rho
    if(nx>0)
      parm$beta<-betahat(y,x,W1a,W2a)
    #Estimate sigma | beta, rho
    parm$sigmasq<-sigmasqhat(muhat(y,x,W1a,W2a,parm$beta))
    #If networks were given, (and not final) estimate rho | beta, sigma
    if(!(final||(nw1+nw2==0))){
      rho<-c(parm$rho1,parm$rho2)
      temp<-optim(rho,n2ll.rho,method=method,control=control,beta=parm$beta, sigmasq=parm$sigmasq)
      if(nw1>0)
        parm$rho1<-temp$par[1:nw1]
      if(nw2>0)
        parm$rho2<-temp$par[(nw1+1):(nw1+nw2)]
    }
    #Calculate model deviance
    parm$dev<-n2ll(W1a,W2a,parm$sigmasq)
    #Return the parameter list
    parm
  }
  #Calculate the expected Fisher information matrix for a fitted model
  infomat<-function(parm){     #Numerical version (requires numDeriv)
    requireNamespace('numDeriv')
    locnll<-function(par){
      #Prepare ll elements according to which parameters are present
      if(nw1>0){
        W1a<-diag(n)-agg(W1,par[(nx+1):(nx+nw1)])
        W1ay<-W1a%*%y
        ladetW1a<-log(abs(det(W1a)))
      }else{
        W1ay<-y
        ladetW1a<-0
      }
      if(nw2>0){
        W2a<-diag(n)-agg(W2,par[(nx+nw1+1):(nx+nw1+nw2)])
        tpW2a<-t(W2a)%*%W2a
        ladetW2a<-log(abs(det(W2a)))
      }else{
        tpW2a<-diag(n)
        ladetW2a<-0
      }
      if(nx>0){
        Xb<-x%*%par[1:nx]
      }else{
        Xb<-0
      }
      #Compute and return
      n/2*(log(2*pi)+log(par[m]))+ t(W1ay-Xb)%*%tpW2a%*%(W1ay-Xb)/(2*par[m]) -ladetW1a-ladetW2a
    }
    #Return the information matrix
    numDeriv::hessian(locnll,c(parm$beta,parm$rho1,parm$rho2,parm$sigmasq)) 
  }
  #How many data points are there?
  n<-length(y)
  #Fix x, W1, and W2, if needed, and count predictors
  if(!is.null(x)){
    if(is.vector(x))
      x<-as.matrix(x)
    if(NROW(x)!=n)
      stop("Number of observations in x must match length of y.")
    nx<-NCOL(x)
  }else
    nx<-0
  if(!is.null(W1)){
    W1<-as.sociomatrix.sna(W1)
    if(!(is.matrix(W1)||is.array(W1)))
      stop("All networks supplied in W1 must be of identical order.")
    if(dim(W1)[2]!=n)
      stop("Order of W1 must match length of y.")
    if(length(dim(W1))==2)
      W1<-array(W1,dim=c(1,n,n))
    nw1<-dim(W1)[1]
  }else
    nw1<-0
  if(!is.null(W2)){
    W2<-as.sociomatrix.sna(W2)
    if(!(is.matrix(W2)||is.array(W2)))
      stop("All networks supplied in W2 must be of identical order.")
    if(dim(W2)[2]!=n)
      stop("Order of W2 must match length of y.")
    if(length(dim(W2))==2)
      W2<-array(W2,dim=c(1,n,n))
    nw2<-dim(W2)[1]
  }else
    nw2<-0
   #Determine the computation mode from the x,W1,W2 parameters
   comp.mode<-as.character(as.numeric(1*(nx>0)+10*(nw1>0)+100*(nw2>0)))
   if(comp.mode=="0")
      stop("At least one of x, W1, W2 must be specified.\n")
   #How many predictors?   
   m<-switch(comp.mode,
      "1"=nx+1,
      "10"=nw1+1,
      "100"=nw2+1,
      "11"=nx+nw1+1,
      "101"=nx+nw2+1,
      "110"=nw1+nw2+1,
      "111"=nx+nw1+nw2+1
   )
  #Initialize the parameter list
  parm<-list()
  if(is.null(theta.seed)){
    if(nx>0)
      parm$beta<-rep(0,nx)
    if(nw1>0)
      parm$rho1<-rep(0,nw1)
    if(nw2>0)
      parm$rho2<-rep(0,nw2)
    parm$sigmasq<-1
  }else{
    if(nx>0)
      parm$beta<-theta.seed[1:nx]
    if(nw1>0)
      parm$rho1<-theta.seed[(nx+1):(nx+nw1)]
    if(nw2>0)
      parm$rho2<-theta.seed[(nx+nw1+1):(nx+nw1+nw2)]
    parm$sigmasq<-theta.seed[nx+nw1+nw2+1]
  }
  parm$dev<-Inf
  #Fit the model
  olddev<-Inf
  while(is.na(parm$dev-olddev)||(abs(parm$dev-olddev)>tol)){
    olddev<-parm$dev
    parm<-estimate(parm,final=FALSE)
  }
  parm<-estimate(parm,final=TRUE)  #Final refinement
  #Assemble the result
  o<-list()
  o$y<-y
  o$x<-x
  o$W1<-W1
  o$W2<-W2
  o$model<-comp.mode
  o$infomat<-infomat(parm)
  o$acvm<-qr.solve(o$infomat)
  o$null.model<-match.arg(null.model)
  o$lnlik.null<-switch(match.arg(null.model),  #Fit a null model
    "meanstd"=sum(dnorm(y-mean(y),0,as.numeric(sqrt(var(y))),log=TRUE)),
    "mean"=sum(dnorm(y-mean(y),log=TRUE)),
    "std"=sum(dnorm(y,0,as.numeric(sqrt(var(y))),log=TRUE)),
    "none"=sum(dnorm(y,log=TRUE))
  )
  o$df.null.resid<-switch(match.arg(null.model),  #Find residual null df
    "meanstd"=n-2,
    "mean"=n-1,
    "std"=n-1,
    "none"=n
  )
  o$df.null<-switch(match.arg(null.model),  #Find null df
    "meanstd"=2,
    "mean"=1,
    "std"=1,
    "none"=0
  )
  o$null.param<-switch(match.arg(null.model),  #Find null params, if any
    "meanstd"=c(mean(y),sqrt(var(y))),
    "mean"=mean(y),
    "std"=sqrt(var(y)),
    "none"=NULL
  )
  o$lnlik.model<--parm$dev/2
  o$df.model<-m
  o$df.residual<-n-m
  o$df.total<-n
  o$beta<-parm$beta                       #Extract parameters
  o$rho1<-parm$rho1  
  o$rho2<-parm$rho2  
  o$sigmasq<-parm$sigmasq
  o$sigma<-o$sigmasq^0.5
  temp<-sqrt(diag(o$acvm))                #Get standard errors
  if(nx>0)
    o$beta.se<-temp[1:nx]
  if(nw1>0)
    o$rho1.se<-temp[(nx+1):(nx+nw1)]
  if(nw2>0)
    o$rho2.se<-temp[(nx+nw1+1):(nx+nw1+nw2)]
  o$sigmasq.se<-temp[m]
  o$sigma.se<-o$sigmasq.se^2/(4*o$sigmasq)  #This a delta method approximation
  if(!is.null(o$beta)){                   #Set X names
    if(!is.null(colnames(x))){
       names(o$beta)<-colnames(x)
       names(o$beta.se)<-colnames(x)
    }else{
       names(o$beta)<-paste("X",1:nx,sep="")
       names(o$beta.se)<-paste("X",1:nx,sep="")
    }
  }
  if(!is.null(o$rho1)){                     #Set W1 names
    if((!is.null(dimnames(W1)))&&(!is.null(dimnames(W1)[[1]]))){
       names(o$rho1)<-dimnames(W1)[[1]]
       names(o$rho1.se)<-dimnames(W1)[[1]]
    }else{
       names(o$rho1)<-paste("rho1",1:nw1,sep=".")
       names(o$rho1.se)<-paste("rho1",1:nw1,sep=".")
    }
  }
  if(!is.null(o$rho2)){                     #Set W2 names
    if((!is.null(dimnames(W2)))&&(!is.null(dimnames(W2)[[1]]))){
       names(o$rho2)<-dimnames(W2)[[1]]
       names(o$rho2.se)<-dimnames(W2)[[1]]
    }else{
       names(o$rho2)<-paste("rho2",1:nw2,sep=".")
       names(o$rho2.se)<-paste("rho2",1:nw2,sep=".")
    }
  }
  if(nw1>0)                               #Aggregate W1 weights
     W1ag<-agg(W1,o$rho1)
  if(nw2>0)                               #Aggregate W2 weights
     W2ag<-agg(W2,o$rho2)
  o$disturbances<-as.vector(switch(comp.mode,  #The estimated disturbances
    "1"=y-x%*%o$beta,
    "10"=(diag(n)-W1ag)%*%y,
    "100"=(diag(n)-W2ag)%*%y,
    "11"=(diag(n)-W1ag)%*%y-x%*%o$beta,
    "101"=(diag(n)-W2ag)%*%(y-x%*%o$beta),
    "110"=(diag(n)-W2ag)%*%((diag(n)-W1ag)%*%y),
    "111"=(diag(n)-W2ag)%*%((diag(n)-W1ag)%*%y-x%*%o$beta)
  ))
  o$fitted.values<-as.vector(switch(comp.mode,  #Compute the fitted values
    "1"=x%*%o$beta,
    "10"=rep(0,n),
    "100"=rep(0,n),
    "11"=qr.solve(diag(n)-W1ag,x%*%o$beta),
    "101"=x%*%o$beta,
    "110"=rep(0,n),
    "111"=qr.solve(diag(n)-W1ag,x%*%o$beta)
  ))
  o$residuals<-as.vector(y-o$fitted.values)
  o$call<-match.call()
  class(o)<-c("lnam")
  o
}
