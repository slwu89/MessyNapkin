# mle urchins
urchins <- read.table("https://webhomes.maths.ed.ac.uk/~swood34/data/urchin-vol.txt")

v0e <- expression(-log(2*pi*sigma^2)/2 -
     (sqrt(y) - sqrt(
          exp(w)*exp(exp(g)*a)
     ))^2/(2*sigma^2)
     - log(2*pi) - log(sig.g*sig.p) -
     (g-mu.g)^2/(2*sig.g^2) - (p-mu.p)^2/(2*sig.p^2))

v0 <- deriv(v0e,c("g","p"), hessian=TRUE,function.arg=
     c("a","y","g","p","w","mu.g","sig.g","mu.p",
                                    "sig.p","sigma"))

v1e <- expression(-log(2*pi*sigma^2)/2 -
     (sqrt(y) - sqrt(
          exp(p)/exp(g) + exp(p)*(a - log(exp(p) / (exp(g)*exp(w)))/exp(g))          
     ))^2/(2*sigma^2)
     - log(2*pi) - log(sig.g*sig.p) -
     (g-mu.g)^2/(2*sig.g^2) - (p-mu.p)^2/(2*sig.p^2))

v1 <- deriv(v1e,c("g","p"), hessian=TRUE,function.arg=
     c("a","y","g","p","w","mu.g","sig.g","mu.p",
                                    "sig.p","sigma"))


pdR <- function(H,k.mult=20,tol=.Machine$double.eps^.8) {
     k <- 1
     tol <- tol * norm(H)
     n <- ncol(H)
     while (
          inherits(try(R <- chol(H + (k-1)*tol*diag(n)), silent=TRUE),"try-error")
     ) {
          k <- k * k.mult
     }
     R 
}

# eval \log f_{\theta}(y,b) and the gradient and hessian
lfyb <- function(b,y,a,th) {
     ## evaluate joint p.d.f. of y and b + grad. and Hessian.
     n <- length(y)
     g <- b[1:n]; p <- b[1:n+n]
     am <- (p-g-th[1])/exp(g)
     ind <- a < am
     f0 <- v0(a[ind],y[ind],g[ind],p[ind],
          th[1],th[2],th[3],th[4],th[5],th[6])
     f1 <- v1(a[!ind],y[!ind],g[!ind],p[!ind],
          th[1],th[2],th[3],th[4],th[5],th[6])
     lf <- sum(f0) + sum(f1)
     g <- matrix(0,n,2) ## extract gradient to g... 
     g[ind,] <- attr(f0,"gradient") ## dlfyb/db 
     g[!ind,] <- attr(f1,"gradient") ## dlfyb/db
     h <- array(0,c(n,2,2)) ## extract Hessian to H... 
     h[ind,,] <- attr(f0,"hessian")
     h[!ind,,] <- attr(f1,"hessian") 
     H <- matrix(0,2*n,2*n)
     for (i in 1:2) for (j in 1:2) {
          indi <- 1:n + (i-1)*n; indj <- 1:n + (j-1)*n
          diag(H[indi,indj]) <- h[,i,j]
     }
     list(lf=lf,g=as.numeric(g),H=H)
}

# s \log f_{\theta}(y,b) + \log f_{\theta^{'}}(y,b)
lfybs <- function(s,b,vol,age,th,thp) {
     ## evaluate s log f(y,b;th) + log f(y,b;thp)
     lf <- lfyb(b,vol,age,thp)
     if (s!=0) {
          lfs <- lfyb(b,vol,age,th)
          lf$lf <- lf$lf + s * lfs$lf
          lf$g <- lf$g + s * lfs$g
          lf$H <- lf$H + s * lfs$H
     }
     lf 
}

# now need function to maximize this wrt b
# returns \log f_{\theta}(y, \hat{b}) (just the joint log likelihood at \hat{b}) if s=0
# and \log |H_{s}|/s otherwise to compute 5.14 which is the approximation of Q_{\theta^{'}}(\theta)
laplace <- function(s=0,th,thp,vol,age,b=NULL,tol=.Machine$double.eps^.7) {
     ii <- c(3,5,6) # variance parameters
     thp[ii] <- exp(thp[ii])
     th[ii] <- exp(th[ii])
     n <- length(vol)
     # initialize b
     if (is.null(b)) {
          b <- c(rep(thp[2],n),rep(thp[4],n))
     } 
     lf <- lfybs(s,b,vol,age,th,thp)
     # newton loop to find \hat{b}
     for (i in 1:200) {
          # R'R = fixed Hessian, R upper tri
          R <- pdR(-lf$H)
          step <- backsolve(R,forwardsolve(t(R),lf$g)) ## Newton 
          conv <- ok <- FALSE
          while (!ok) {
               lf1 <- lfybs(s,b+step,vol,age,th,thp)
               if (sum(abs(lf1$g)>abs(lf1$lf)*tol)==0 || sum(b+step!=b)==0) {
                    conv <- TRUE
               }
               kk <- 0
               if (!conv&&kk<30&&(!is.finite(lf1$lf) || lf1$lf < lf$lf)) {
                    step <- step/2;kk <- kk+1
               } else {
                    ok <- TRUE
               }
          }
          dlf <- abs(lf$lf-lf1$lf);lf <- lf1;b <- b + step;
          if (dlf<tol*abs(lf$lf)||conv||kk==30) {
               break
          }
     }
     if (s==0) {
          return(list(g=lfyb(b,vol,age,th)$lf,b=b))
     }
     R <- pdR(-lf$H,10)
     list(b=b,rldetH = sum(log(diag(R))))
}


Q <- function(th,thp,vol,age,eps=1e-5) {
     ## 1. find b.hat maximising log joint density at thp
     if (exists(".inib",envir=environment(Q))) { 
          b <- get(".inib",envir=environment(Q))
     } else {
          b <- NULL
     }
     la <- laplace(s=0,th,thp,vol,age,b=b)
     assign(".inib",la$b,envir=environment(Q))
     ## 2. For s = -eps and eps find b maximising s log joint ## at th + log joint at thp along with log|H_s|.
     lap <- laplace(s=eps/2,th,thp,vol,age,b=la$b)$rldetH
     lam <- laplace(s= -eps/2,th,thp,vol,age,b=la$b)$rldetH
     la$g - (lap-lam)/eps
}

thp <- th <- rep(0,6); ## starting values
for (i in 1:30) { ## EM loop
     er <- optim(th,Q,control=list(fnscale=-1,maxit=200),vol=urchins$vol,age=urchins$age,thp=thp)
     th <- thp <- er$par
     cat(th,"\n")
}


# test within Q
urchin_sm <- urchins[c(1,25,50,75,100,115,125,142), ]
th <- c(
    -3.0,
    -0.3, 
    -1.5,
    0.15,
    -1.5,
    -1.37
)
thp <- rep(0, 6)
n <- nrow(urchin_sm)
b <- c(rep(th[2],n),rep(th[4],n))

la <- laplace(s=0,th,thp,vol=urchin_sm$vol,age=urchin_sm$age,b=b)
la

lfyb(la$b,urchin_sm$vol,urchin_sm$age,th)$lf


eps = 1e-5
lap <- laplace(s=eps/2,th,thp,vol=urchin_sm$vol,age=urchin_sm$age,b=b)$rldetH
lam <- laplace(s= -eps/2,th,thp,vol=urchin_sm$vol,age=urchin_sm$age,b=b)$rldetH


laplace_test <- function(s=0,th,thp,vol,age,b=NULL,tol=.Machine$double.eps^.7) {
     ii <- c(3,5,6) # variance parameters
     thp[ii] <- exp(thp[ii])
     th[ii] <- exp(th[ii])
     n <- length(vol)
     # initialize b
     if (is.null(b)) {
          b <- c(rep(thp[2],n),rep(thp[4],n))
     } 
     lf <- lfybs(s,b,vol,age,th,thp)
     # newton loop to find \hat{b}
     for (i in 1:200) {
          # R'R = fixed Hessian, R upper tri
          R <- pdR(-lf$H)
          step <- backsolve(R,forwardsolve(t(R),lf$g)) ## Newton 
          conv <- ok <- FALSE
          while (!ok) {
               lf1 <- lfybs(s,b+step,vol,age,th,thp)
               if (sum(abs(lf1$g)>abs(lf1$lf)*tol)==0 || sum(b+step!=b)==0) {
                    conv <- TRUE
               }
               kk <- 0
               if (!conv&&kk<30&&(!is.finite(lf1$lf) || lf1$lf < lf$lf)) {
                    step <- step/2;kk <- kk+1
               } else {
                    ok <- TRUE
               }
          }
          dlf <- abs(lf$lf-lf1$lf);lf <- lf1;b <- b + step;
          if (dlf<tol*abs(lf$lf)||conv||kk==30) {
               break
          }
     }
     if (s==0) {
          return(list(g=lfyb(b,vol,age,th)$lf,b=b))
     }
     R <- pdR(-lf$H,10)
     list(b=b,rldetH = sum(log(diag(R))), H=lf$H, R=R)
}

lap <- laplace_test(s=eps/2,th,thp,vol=urchin_sm$vol,age=urchin_sm$age,b=b)

norm(lap$H)

pdR <- function(H,k.mult=20,tol=.Machine$double.eps^.8) {
     k <- 1
     tol <- tol * norm(H)
     n <- ncol(H)
     while (
          inherits(try(R <- chol(H + (k-1)*tol*diag(n)), silent=TRUE),"try-error")
     ) {
          k <- k * k.mult
     }
     R 
}

lap_H <- as.matrix(read.csv('./Statistics/Urchin/H.csv', row.names=NULL, header=FALSE))

pdR(lap_H)

colnames(lap_H) <- NULL

tol=.Machine$double.eps^.8
k <- 1
n <- ncol(lap_H)

chol(lap_H + (k-1)*tol*diag(n))

inherits(try(R <- chol(H + (k-1)*tol*diag(n)), silent=TRUE),"try-error")

m <- matrix(c(5,1,1,3),2,2)
chol(m)
