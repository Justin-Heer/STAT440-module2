library(mice)
library(dplyr)
library("tidyverse")
library("car") 
library("rpart") 
library("rpart.plot") 
library("nnet") 
library("randomForest")
library("effects")
library(mgcv)
library(caret)
library(MASS) # for ridge
library(glmnet)
library(rgl) 
library(pls)  
library(tidyselect)
Xtrain<- read.table("Xtrain.txt",header = TRUE)
Ytrain<- read.table("Ytrain.txt",header = TRUE)
Xtest=read.table("Xtest.txt",header = TRUE)
Ytest=read.table("Ytest.txt",header = TRUE)

# impute=mice(train_X1000,m=5,seed=123)
# impute
# impute$imp$B01

source("Helper Functions.R")
#checking missing values
#variable.summary(Xtrain)

#filling missing values with mean for Xtrain
Xtrain2=Xtrain
Xtrain2[Xtrain2 == "?" ] = NA
Xtrain2$G01=as.factor(Xtrain2$G01)
Xtrain2$G02=as.factor(Xtrain2$G02)
Xtrain2$G03=as.factor(Xtrain2$G03)

Xtrain2$B15=as.numeric(as.character(Xtrain2$B15))

for(i in 1:ncol(Xtrain2)){
  Xtrain2[is.na(Xtrain2[,i]), i] <- mean(Xtrain2[,i], na.rm = TRUE)
}

#variable.summary(Xtrain2)

#filling missing values with mean for Ytrain
variable.summary(Ytrain)
Ytrain2=Ytrain
Ytrain2$Z02=as.factor(Ytrain2$Z02)
for(i in 1:ncol(Ytrain2)){
  Ytrain2[is.na(Ytrain2[,i]), i] <- mean(Ytrain2[,i], na.rm = TRUE)
}

#variable.summary(Ytrain2)

#filling missing values with mean for Xtest
variable.summary(Xtest)
Xtest2=Xtest
Xtest2$G01=as.factor(Xtest2$G01)
Xtest2$G02=as.factor(Xtest2$G02)
Xtest2$G03=as.factor(Xtest2$G03)


Xtest2[Xtest2 == "?" ] = NA
Xtest2$B15=as.numeric(as.character(Xtest2$B15))


for(i in 1:ncol(Xtest2)){
  Xtest2[is.na(Xtest2[,i]), i] <- mean(Xtest2[,i], na.rm = TRUE)
}

# rownames(Xtrain2)=Xtrain2$Id
# Xtrain2=dplyr::select(Xtrain2,-c("Id"))
# rownames(Ytrain2)=Ytrain2$Id
# Ytrain2=dplyr::select(Ytrain2,-c("Id"))
# rownames(Xtest2)=Xtest2$Id
# Xtest2=dplyr::select(Xtest2,-c("Id"))
Full_data=merge(x=Xtrain2,y=Ytrain2,by="Id")
#variable.summary(Full_data)

custom =trainControl(method="repeatedcv",
                     number=10,
                     repeats=1,
                     verboseIter = T)

rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
}

##############################################################################

#Z14
train14=dplyr::select(Full_data,-c("Id","Z02","Z03", "Z04", "Z05", "Z06", "Z07", "Z08",
                                   "Z09", "Z10", "Z11", "Z12", "Z13", "Z01","G01","G02","G03","F09","F10","F11","F12"))
test14=dplyr::select(Xtest2,-c("Id","G01","G02","G03","F09","F10","F11","F12"))
tuned.nnet14 <- train(Z14~.,train14, method="nnet",
                      trace=FALSE, linout=TRUE,
                      trControl=custom, preProcess="range",
                      tuneGrid = expand.grid(size=c(15,16,17),decay=c(0.5,1,2)))
predZ14=predict(tuned.nnet14,test14)

X.train.raw =train14[,-56]
X.train = rescale(X.train.raw, X.train.raw)
Y.train = train14$Z14

X.valid.raw = test14
X.valid = rescale(X.valid.raw, X.train.raw)


# fit.nnet = nnet(y = Y.train, x = X.train, linout = TRUE, size = 17,
#                 decay = 1, maxit = 500)


fit.nnet14=nnet(y = Y.train, x = X.train, linout = TRUE, size = 16,
                decay = 0.5, maxit = 500)

pred.nnet14 = predict(fit.nnet14, X.valid)

predZ14=pred.nnet14