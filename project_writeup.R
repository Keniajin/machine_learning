library(caret)
library(rattle)
library(randomForest)

#creatind the data folder if it doesnt exist
if (!file.exists("data")) {
  dir.create("data")
}

#downloadTheData
if (!file.exists("data/pml-training.csv")){
  fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
  download.file(fileUrl, destfile = "data/pml-training.csv")
}else {message("Data Already downloaded")}

if (!file.exists("data/pml-testing.csv")){
  fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
  download.file(fileUrl, destfile = "data/pml-testing.csv")
}else {message("Data Already downloaded")}


#load the data to R
pml_training <- read.table("data/pml-training.csv", header = T , sep = ",")
pml_testing <-  read.table("data/pml-testing.csv", header = T , sep = ",")


set.seed(975)

#colnames in training data
colnames_training <- colnames(pml_training)
colnames_predictors <- colnames_training[ grep(".*belt.*|.*arm.*|.*dumbbell.*",
                                              colnames(pml_training)) ]

cols_no_NA <- sapply(pml_training[,colnames_predictors],
                    function(x) !any(is.na(x)))

# reduce predictors to columns free of NA values
colnames_predictors <- colnames_predictors[cols_no_NA]

# eliminate predictor columns with "#DIV/0!"
cols_no_DIV0 <- sapply(pml_training[,colnames_predictors],
                      function(x) length(grep(".*DIV/0.*",x))==0)
colnames_predictors <- colnames_predictors[cols_no_DIV0]

#the count of values with predictors
length(colnames_predictors)


# dataframes containing only the intended predictors
pmlTrain <- pml_training[,colnames_predictors]
pmlTrain$classe <- pml_training$classe
pmlTest <-  pml_testing[,c(colnames_predictors)]
pmlTest$classe <- pml_testing$classe

#in training data create a training and testing data
inTrain <- createDataPartition(y=pmlTrain$classe,
                               p=0.6, list=FALSE)
pmlTrain_train <- pmlTrain[inTrain,]
pmlTest_test <- pmlTrain[-inTrain,]

#columns related to be used as predictors
is_Roll <- (substr(names(pmlTrain_train),1,4) == "roll")
##is_Belt <- (substr(names(pmlTrain_train),1,4) == "belt")
is_Yaw <- (substr(names(pmlTrain_train),1,3) == "yaw")
is_Pitch <- (substr(names(pmlTrain_train),1,5) == "pitch")


#featurePlot(x=pmlTrain_train[,names(pmlTrain_train)[is_Roll]], y= pmlTrain_train$classe, plot="pairs")
##featurePlot(x=pmlTrain_train[,names(pmlTrain_train)[is_Belt]], y= pmlTrain_train$classe, plot="pairs")
#featurePlot(x=pmlTrain_train[,names(pmlTrain_train)[is_Yaw]], y= pmlTrain_train$classe, plot="pairs")

#create a classification tree
modFit1 <- train(classe ~ .,method="rpart",data=pmlTrain_train)
print(modFit1$finalModel)

## Plot tree
plot(modFit1$finalModel, uniform=TRUE, 
     main="Classification Tree")
text(modFit1$finalModel, use.n=TRUE, all=TRUE, cex=.8)

## Prettier plots
##not clear
fancyRpartPlot(modFit1$finalModel)

### estimate the prediction
pred1 <- predict(modFit1,newdata=pmlTrain_train)
tab1 <- table(pmlTrain_train$classe,pred1)

## miss classification rate
1-sum(diag(tab1))/sum(tab1)

# create a Random Forest model
modFit2 <- randomForest(classe ~ ., data=pmlTrain_train)
# display model fit results
modFit2

# Generic plot using model from randomForest
plot(modFit2, log="y",
     main="Estimated Out-of-Bag (OOB) Error and Class Error of Random Forest Model")
legend("top", colnames(modFit2$err.rate), col=1:6, cex=0.8, fill=1:6)


# Dotchart of variable importance as measured by randomForest
varImpPlot(modFit2, main="Variable Importance in the Random Forest Model")
#plot1 <- varImp(modFit2, scale = FALSE)
#plot(plot1, top = 20)


# estimate out of sample error
pred_out_of_sample <- predict(modFit2, newdata=subset(pmlTest_test, select=-classe))

# out of sample confusion matrix
confusion_matrix <- table(pred_out_of_sample, pmlTest_test$classe)
confusion_matrix
# estimated out of sample error rate
out_of_sample_error_rate = 1.00 - sum(diag(confusion_matrix)) / sum(confusion_matrix)
out_of_sample_error_rate

pred2 <- predict(modFit2,newdata=pmlTest_test)
tab2 <- table(pmlTest_test$classe,pred2)
tab2

## miss classification rate
1-sum(diag(tab2))/sum(tab2)

# predict using pml-test data
pred <- predict(modFit2, newdata=pml_testing)
pred
