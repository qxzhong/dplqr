library(latex2exp)
par(mfrow=c(2,2))
## DPLQR
res <- read.csv(file="../data/resDeep.csv",header=TRUE)
tau <- res$tau
est <- res$estCoef
ci_lower <- res$lowerCI
ci_upper <- res$upperCI
CLDeep <- res$CL
# CI at different quantile levels
plot(tau,ci_lower,type="l",lty=3,col='gray20',ylim=c(min(ci_lower),max(ci_upper )),
     xlab=TeX('Quantile level $\\tau$'),ylab=TeX("$\\hat{\\theta}_{\\tau}$"),
     main=TeX("Estimated coefficient $\\hat{\\theta}_{\\tau}$ using DPLQR"))
lines(tau,ci_upper,type="l",lty=3,
      col='gray20')
polygon(c(tau,rev(tau)),c(ci_lower,rev(ci_upper)),
        col='gray60', lty = 0)
abline(h=0,col="gray50")
lines(tau,est,col="gray10")

## LQR
res <- read.csv(file="../data/resLin.csv",header=TRUE)
tau <- res$tau
est <- res$estCoef
ci_lower <- res$lowerCI
ci_upper <- res$upperCI
CLLQR <- res$CL
# CI at different quantile levels
plot(tau,ci_lower,type="l",lty=3,col='gray20',ylim=c(min(ci_lower),max(ci_upper )),
     xlab=TeX('Quantile level $\\tau$'),ylab=TeX("$\\hat{\\theta}_{\\tau}$"),
     main=TeX("Estimated coefficient $\\hat{\\theta}_{\\tau}$ using LQR"))
lines(tau,ci_upper,type="l",lty=3,
      col='gray20')
polygon(c(tau,rev(tau)),c(ci_lower,rev(ci_upper)),
        col='gray60', lty = 0)
abline(h=0,col="gray50")
lines(tau,est,col="gray10")

## PLAQR
res <- read.csv(file="../data/resAdd.csv",header=TRUE)
tau <- res$tau
est <- res$estCoef
ci_lower <- res$lowerCI
ci_upper <- res$upperCI
CLAdd <- res$CL
# CI at different quantile levels
plot(tau,ci_lower,type="l",lty=3,col='gray20',ylim=c(min(ci_lower),max(ci_upper )),
     xlab=TeX('Quantile level $\\tau$'),ylab=TeX("$\\hat{\\theta}_{\\tau}$"),
     main=TeX("Estimated coefficient $\\hat{\\theta}_{\\tau}$ using PLAQR"))
lines(tau,ci_upper,type="l",lty=3,
      col='gray20')
polygon(c(tau,rev(tau)),c(ci_lower,rev(ci_upper)),
        col='gray60', lty = 0)
abline(h=0,col="gray50")
lines(tau,est,col="gray10")


# prediction (w.r.t. check loss) on test set
plot(tau,CLDeep,type="l",lty=1,lwd=2,col="red",
     xlab=TeX('Quantile level $\\tau$'),
     ylab="Check loss on test set",
     ylim=c(min(CLDeep),max(CLDeep)+0.05),
     main="Prediction performance")
lines(tau,CLLQR,col="blue",lty=2,lwd=2)
lines(tau,CLAdd,col="green",lty=3,lwd=2)


