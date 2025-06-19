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

# eval \log f_{\theta}(y,b) and the gradient and hessian
# note this assumes that variance params in th are already exp() transformed
# back to constrained scale
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

# theta on log (unconstrained) scale
llu<-function(theta,vol,age,tol=.Machine$double.eps^.8){
## Laplace approximate log likelihood for urchin model.
    ii <- c(3,5,6)
    theta[ii] <- exp(theta[ii]) ## variance params 
    n <- length(vol)
    if (exists(".inib",envir=environment(llu))) {
        b <- get(".inib",envir=environment(llu))
    } else b <- c(rep(theta[2],n),rep(theta[4],n)); ## init 
    lf <- lfyb(b,vol,age,theta)
    for (i in 1:200) { ## Newton loop...
        R <- pdR(-lf$H) ## R’R = (perturbed) Hessian
        step <- backsolve(R,forwardsolve(t(R),lf$g)) ## Newton 
        conv <- ok <- FALSE
        while (!ok) { ## step halving
            lf1 <- lfyb(b+step,vol,age,theta);
            if (sum(abs(lf1$g)>abs(lf1$lf)*tol)==0) conv <- TRUE
            kk <- 0
            if (!conv&&kk<30&&
                (!is.finite(lf1$lf) || lf1$lf < lf$lf)) {
            step <- step/2;kk <- kk+1
            } else ok <- TRUE
        }
        lf <- lf1;b <- b + step
        if (kk==30||conv) break ## if converged or failed 
    } ## end of Newton loop 
    assign(".inib",b,envir=environment(llu))
    R <- pdR(-lf$H,10)
    ll <- lf$lf - sum(log(diag(R))) + log(2*pi)*n
    -ll 
}

th <- c(-4,-.2,log(.1),.2,log(.1),log(.5)) ## initial

fit <- optim(th,llu,method="BFGS",vol=urchins$vol, age=urchins$age,hessian=TRUE)
2*fit$value + 2*length(fit$par) ## AIC
fit$par
