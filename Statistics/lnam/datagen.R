# rm(list=ls());gc();dev.off()
# library(sna)
# set.seed(4395834L)

# n <- 100

# w1<-rgraph(n)               #Draw the AR matrix
# w2<-w1               #Draw the MA matrix
# x<-matrix(rnorm(n*5),n,5) #Draw some covariates
# r1<-0.2                       #Set the model parameters
# r2<-0.1
# sigma<-0.1
# beta<-rnorm(5)
# #Assemble y from its components:
# nu<-rnorm(n,0,sigma)          #Draw the disturbances
# e<-qr.solve(diag(n)-r2*w2,nu) #Draw the effective errors
# y<-qr.solve(diag(n)-r1*w1,x%*%beta+e)  #Compute y

library(sna)
library(numDeriv)

# write info
write.csv(file="./data/small/w1.csv", x=w1, row.names=FALSE)
write.csv(file="./data/small/w2.csv", x=w2, row.names=FALSE)
write.csv(file="./data/small/x.csv", x=x, row.names=FALSE)
write.csv(file="./data/small/beta.csv", x=beta, row.names=FALSE)
write.csv(file="./data/small/nu.csv", x=nu, row.names=FALSE)
write.csv(file="./data/small/e.csv", x=e, row.names=FALSE)
write.csv(file="./data/small/y.csv", x=y, row.names=FALSE)

# # testing
# w1 <- as.matrix(read.csv(file="./data/w1.csv"))
# dimnames(w1) <- list(NULL,NULL)
# w2 <- as.matrix(read.csv(file="./data/w2.csv"))
# dimnames(w2) <- list(NULL,NULL)
# x <- as.matrix(read.csv(file="./data/x.csv"))
# dimnames(x) <- list(NULL,NULL)

# beta <- read.csv(file="./data/beta.csv")$x
# nu <- read.csv(file="./data/nu.csv")$x
# e <- read.csv(file="./data/e.csv")$x
# y <- read.csv(file="./data/y.csv")$V1

# # run through the function
# W1 <- w1
# W2 <- w2

# #How many data points are there?
# n<-length(y)
# #Fix x, W1, and W2, if needed, and count predictors
# nx<-NCOL(x)
# W1<-array(W1,dim=c(1,n,n))
# nw1<-dim(W1)[1]

# W2<-array(W2,dim=c(1,n,n))
# nw2<-dim(W2)[1]

# #Determine the computation mode from the x,W1,W2 parameters
# comp.mode<-as.character(as.numeric(1*(nx>0)+10*(nw1>0)+100*(nw2>0)))

# #How many predictors?   
# m<-switch(comp.mode,
#     "1"=nx+1,
#     "10"=nw1+1,
#     "100"=nw2+1,
#     "11"=nx+nw1+1,
#     "101"=nx+nw2+1,
#     "110"=nw1+nw2+1,
#     "111"=nx+nw1+nw2+1
# )

# parm<-list()
# parm$beta<-rep(0,nx)
# parm$rho1<-rep(0,nw1)
# parm$rho2<-rep(0,nw2)
# parm$sigmasq<-1
# parm$dev<-Inf

# tol <- 1e-10
# method = "BFGS"
# control = list()

# # --------------------------------------------------------------------------------
# # debug iterations
# parm<-estimate(parm,final=FALSE)
# parm<-estimate(parm,final=FALSE)
# parm<-estimate(parm,final=FALSE)

# # --------------------------------------------------------------------------------
# # by hand
# # an iteration of estimate
# W1a<-diag(n)-agg(W1,parm$rho1)
# W2a<-diag(n)-agg(W2,parm$rho2)

# # assume covariates given ,estimate beta | rho
# tXtW2aW2a<-t(x)%*%t(W2a)%*%W2a
# parm$beta <- qr.solve(tXtW2aW2a%*%x,tXtW2aW2a%*%W1a%*%y)

# #Estimate sigma | beta, rho
# parm$sigmasq<-sigmasqhat(muhat(y,x,W1a,W2a,parm$beta))

# #If networks were given, (and not final) estimate rho | beta, sigma
# n2ll.rho(c(parm$rho1,parm$rho2),parm$beta,parm$sigmasq)

# if(!(final||(nw1+nw2==0))){
#   rho<-c(parm$rho1,parm$rho2)
#   temp<-optim(rho,n2ll.rho,method="BFGS",control=list(),beta=parm$beta, sigmasq=parm$sigmasq)
#   if(nw1>0)
#     parm$rho1<-temp$par[1:nw1]
#   if(nw2>0)
#     parm$rho2<-temp$par[(nw1+1):(nw1+nw2)]
# }

# #Calculate model deviance
# parm$dev<-n2ll(W1a,W2a,parm$sigmasq)


# # --------------------------------------------------------------------------------
# #Fit the model
# parm<-list()
# parm$beta<-rep(0,nx)
# parm$rho1<-rep(0,nw1)
# parm$rho2<-rep(0,nw2)
# parm$sigmasq<-1
# parm$dev<-Inf

# tol <- 1e-10
# method = "BFGS"
# control = list()

# olddev<-Inf
# i <- 0
# while(is.na(parm$dev-olddev)||(abs(parm$dev-olddev)>tol)){
#     olddev<-parm$dev
#     parm<-estimate(parm,final=FALSE)
#     i <- i + 1
# }
# parm<-estimate(parm,final=TRUE)  #Final refinement

# # get info matrix
# mod_infomat<-infomat(parm)
# mod_acvm<-qr.solve(mod_infomat)
# mod_se <- sqrt(diag(mod_acvm))


# # functions from lnam
# #Define the log-likelihood functions for each case
# agg<-function(a,w){
#   m<-length(w)
#   n<-dim(a)[2]
#   mat<-as.double(matrix(0,n,n))
#   matrix(.C("aggarray3d_R",as.double(a),as.double(w),mat=mat,as.integer(m), as.integer(n),PACKAGE="sna",NAOK=TRUE)$mat,n,n)
# }
#   #Estimate covariate effects, conditional on autocorrelation parameters
#   betahat<-function(y,X,W1a,W2a){
#     if(nw1==0){
#       if(nw2==0){
#         return(qr.solve(t(X)%*%X,t(X)%*%y))
#       }else{
#         tXtW2aW2a<-t(X)%*%t(W2a)%*%W2a
#         return(qr.solve(tXtW2aW2a%*%X,tXtW2aW2a%*%y))
#       }
#     }else{
#       if(nw2==0){
#         return(qr.solve(t(X)%*%X,t(X)%*%W1a%*%y))
#       }else{
#         tXtW2aW2a<-t(X)%*%t(W2a)%*%W2a
#         qr.solve(tXtW2aW2a%*%X,tXtW2aW2a%*%W1a%*%y)
#       }
#     }
#   }
#   #Estimate predicted means, conditional on other effects
#   muhat<-function(y,X,W1a,W2a,betahat){
#     if(nx>0)
#       Xb<-X%*%betahat
#     else
#       Xb<-0
#     switch((nw1>0)+2*(nw2>0)+1,
#       y-Xb,
#       W1a%*%y-Xb,
#       W2a%*%(y-Xb),
#       W2a%*%(W1a%*%y-Xb)
#     )
#   }
#   #Estimate innovation variance, conditional on other effects
#   sigmasqhat<-function(muhat){
#     t(muhat)%*%muhat/length(muhat)
#   }
#   #Model deviance (for use with fitting rho | beta, sigma)
#   n2ll.rho<-function(rho,beta,sigmasq){
#     #Prepare ll elements according to which parameters are present
#     if(nw1>0){
#       W1a<-diag(n)-agg(W1,rho[1:nw1])
#       W1ay<-W1a%*%y
#       adetW1a<-abs(det(W1a))
#     }else{
#       W1ay<-y
#       adetW1a<-1
#     }
#     if(nw2>0){
#       W2a<-diag(n)-agg(W2,rho[(nw1+1):(nw1+nw2)])
#       tpW2a<-t(W2a)%*%W2a
#       adetW2a<-abs(det(W2a))
#     }else{
#       tpW2a<-diag(n)
#       adetW2a<-1
#     }
#     if(nx>0){
#       Xb<-x%*%beta
#     }else{
#       Xb<-0
#     }
#     #Compute and return
#     n*(log(2*pi)+log(sigmasq)) + t(W1ay-Xb)%*%tpW2a%*%(W1ay-Xb)/sigmasq - 2*(log(adetW1a)+log(adetW2a))
#   }
#   #Model deviance (general purpose)
#   n2ll<-function(W1a,W2a,sigmasqhat){
#     switch((nw1>0)+2*(nw2>0)+1,
#       n*(1+log(2*pi)+log(sigmasqhat)),
#       n*(1+log(2*pi)+log(sigmasqhat))-2*log(abs(det(W1a))),
#       n*(1+log(2*pi)+log(sigmasqhat))-2*log(abs(det(W2a))),
#       n*(1+log(2*pi)+log(sigmasqhat))- 2*(log(abs(det(W1a)))+log(abs(det(W2a))))
#     )
#   }
#   #Conduct a single iterative refinement of a set of initial parameter estimates
#   estimate<-function(parm,final=FALSE){
#     #Either aggregate the weight matrices, or NULL them
#     if(nw1>0)
#       W1a<-diag(n)-agg(W1,parm$rho1)
#     else
#       W1a<-NULL
#     if(nw2>0)
#       W2a<-diag(n)-agg(W2,parm$rho2)
#     else
#       W2a<-NULL
#     #If covariates were given, estimate beta | rho
#     if(nx>0)
#       parm$beta<-betahat(y,x,W1a,W2a)
#     #Estimate sigma | beta, rho
#     parm$sigmasq<-sigmasqhat(muhat(y,x,W1a,W2a,parm$beta))
#     #If networks were given, (and not final) estimate rho | beta, sigma
#     if(!(final||(nw1+nw2==0))){
#       rho<-c(parm$rho1,parm$rho2)
#       temp<-optim(rho,n2ll.rho,method=method,control=control,beta=parm$beta, sigmasq=parm$sigmasq)
#       if(nw1>0)
#         parm$rho1<-temp$par[1:nw1]
#       if(nw2>0)
#         parm$rho2<-temp$par[(nw1+1):(nw1+nw2)]
#     }
#     #Calculate model deviance
#     parm$dev<-n2ll(W1a,W2a,parm$sigmasq)
#     #Return the parameter list
#     parm
#   }


# #Calculate the expected Fisher information matrix for a fitted model
# infomat<-function(parm){     #Numerical version (requires numDeriv)
#   requireNamespace('numDeriv')
#   locnll<-function(par){
#     #Prepare ll elements according to which parameters are present
#     if(nw1>0){
#       W1a<-diag(n)-agg(W1,par[(nx+1):(nx+nw1)])
#       W1ay<-W1a%*%y
#       ladetW1a<-log(abs(det(W1a)))
#     }else{
#       W1ay<-y
#       ladetW1a<-0
#     }
#     if(nw2>0){
#       W2a<-diag(n)-agg(W2,par[(nx+nw1+1):(nx+nw1+nw2)])
#       tpW2a<-t(W2a)%*%W2a
#       ladetW2a<-log(abs(det(W2a)))
#     }else{
#       tpW2a<-diag(n)
#       ladetW2a<-0
#     }
#     if(nx>0){
#       Xb<-x%*%par[1:nx]
#     }else{
#       Xb<-0
#     }
#     #Compute and return
#     n/2*(log(2*pi)+log(par[m]))+ t(W1ay-Xb)%*%tpW2a%*%(W1ay-Xb)/(2*par[m]) -ladetW1a-ladetW2a
#   }
#   #Return the information matrix
#   numDeriv::hessian(locnll,c(parm$beta,parm$rho1,parm$rho2,parm$sigmasq)) 
# }