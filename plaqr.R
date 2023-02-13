library("plaqr")

checkLoss_mean <- function(errors,tau=0.5){
  mat <- cbind(tau*errors,(tau-1)*errors) 
  m <- mean(apply(mat,1,max))
  return(m)
}

df_train <- read.csv(file="../data/df_train.csv")
df_test <- read.csv(file="../data/df_test.csv")
# delete index column
df_train <- df_train[,-1]
df_test <- df_test[,-1]

tau_seq <- seq(0.02,0.98,by=0.02)
resAdd <- matrix(0,length(tau_seq),5)
i=0

for (tau in tau_seq) {
  i=i+1
  addFit <- plaqr(outY~., nonlinVars=~blst+flh+sup+cag+fag+age, data=df_train,tau=tau)
  addFitSum <- summary(addFit)
  estCoef <- addFitSum$coefficients[2,1]
  se <- addFitSum$coefficients[2,2]
  intv <- c(addFitSum$coefficients[2,2],addFitSum$coefficients[2,3])
  
  y_pred <- c(predict(addFit,newdata=df_test))
  y_test <- c(df_test[,8])
  y_err <- matrix(y_test,length(y_test))-matrix(y_pred,length(y_pred)) 
  predErrAdd <- checkLoss_mean(y_err,tau)
  resAdd[i,] <- c(tau,estCoef,intv,predErrAdd)
  print(tau)
  print(resAdd[i,])
}

resAdd <- as.data.frame(resAdd)
colnames(resAdd) <- c('tau','estCoef','lowerCI','upperCI','CL')

# CI Area
delta <- tau_seq[2]-tau_seq[1]
CIArea <- sum(resAdd$upperCI - resAdd$lowerCI)*delta
print(paste('95% CI Area under PLAQR:',CIArea,sep=" ") )

# average check loss
ACL <- sum(resAdd$CL)*delta
print(paste('ACL under PLAQR:',ACL,sep=" ") )

# save results for plot
pathmain = "../data/resAdd"
write.csv(resAdd,file=paste(pathmain,".csv",sep=""))













