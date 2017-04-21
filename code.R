white.url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
white.raw <- read.csv(white.url, header = TRUE, sep = ";")
wine <- white.raw
str(wine)

#Applying Models without Normalization

# To classify all wines into bad, normal and good depending upon whether their quality is less than, equal to, or greater than 6)
wine$taste <- ifelse(wine$quality < 6, 'bad', 'good')
wine$taste[wine$quality == 6] <- 'normal'
wine$taste <- as.factor(wine$taste)

#Separating Dataset into training and testing datasets (60%, 40% split)
set.seed(142)
samp <- sample(nrow(wine), 0.6 * nrow(wine))
train <- wine[samp, ]
test <- wine[-samp, ]

#Building Multiple Linear Regression Model 
linearmodel = lm(quality ~ . -taste, data=train)
predictlinearmodel = predict(linearmodel, newdata=test)
# Computing RMSE (Root Mean Square Error)
RMSE_linear=sqrt(sum((test$quality - predictlinearmodel)^2)/nrow(test))
sprintf("The Value of RMSE for Linear model is %f", RMSE_linear)


#Building Regression Tree Model
library(rpart)
rpartmodel <- rpart(quality ~.-taste , data = train)
predictrpartmodel <- predict(rpartmodel, test)
# Computing RMSE (Root Mean Square Error)
RMSE_rpart=sqrt(sum((test$quality - predictrpartmodel)^2)/nrow(test))
sprintf("The Value of RMSE for Rpart model is %f", RMSE_rpart)


#Building Random Forest Model
library(randomForest)
randomForestmodel <- randomForest(taste ~ . - quality, data = train)
predictrandomForestmodel <- predict(randomForestmodel, newdata = test)
x<-table(test$taste, predictrandomForestmodel)
accuracy= (sum(diag(x)))/sum(x)
sprintf("Accuracy of Random Forest is %f", accuracy)
library(caret)
# Precision - Fractions of correct predicitions for a class
precision = diag(x)/apply(x,2,sum) 
# Recall - Fractions of instances of a class that were correctly predicted
recall = diag(x)/apply(x,1,sum)
f1 = 2*precision*recall/(precision+recall)
data.frame(precision, recall, f1) 

#Building SVM Classification Model 
library(e1071)
svmmodel<-svm(taste ~. -quality, data=train)
predictsvmmodel <- predict(svmmodel ,test)
x<-table(test$taste, predictsvmmodel)
accuracy= (sum(diag(x)))/sum(x)
sprintf("Accuracy of SVM is %f", accuracy)
library(caret)
# Precision - Fractions of correct predicitions for a class
precision = diag(x)/apply(x,2,sum) 
# Recall - Fractions of instances of a class that were correctly predicted
recall = diag(x)/apply(x,1,sum)
f1 = 2*precision*recall/(precision+recall)
data.frame(precision, recall, f1) 



#Applying Models with Normalization and Outlier Removal 
# (No Outliers present in total sulphar dioxide and alcohol features)
wine<-cbind(scale(wine[,1:11]), wine[ ,-c(1:11)]) 
wine <- wine[!abs(wine$fixed.acidity) > 3,]
wine <- wine[!abs(wine$volatile.acidity) > 3,]
wine <- wine[!abs(wine$citric.acid) > 3,]
wine <- wine[!abs(wine$residual.sugar) > 3,]
wine <- wine[!abs(wine$chlorides) > 3,]
wine <- wine[!abs(wine$free.sulfur.dioxide) > 3,]
wine <- wine[!abs(wine$density) > 3,]
wine <- wine[!abs(wine$pH) > 3,]
wine <- wine[!abs(wine$sulphates) > 3,]

#Separating Data into training and testing datasets
set.seed(111)
samp <- sample(nrow(wine), 0.6 * nrow(wine))
train <- wine[samp, ]
test <- wine[-samp, ]

#Building Multiple Linear Regression Model 
linearmodel2 = lm(quality ~ . -taste, data=train)
predictlinearmodel2 = predict(linearmodel2, newdata=test)
# Computing RMSE (Root Mean Square Error)
RMSE_linear2=sqrt(sum((test$quality - predictlinearmodel2)^2)/nrow(test))
sprintf("The Value of RMSE for Linear model is %f", RMSE_linear2)

#Building Regression Tree Model
library(rpart)
rpartmodel2 <- rpart(quality ~.-taste , data = train)
predictrpartmodel2 <- predict(rpartmodel2, test)
# Computing RMSE (Root Mean Square Error)
RMSE_rpart2=sqrt(sum((test$quality - predictrpartmodel2)^2)/nrow(test))
sprintf("The Value of RMSE for Rpart model is %f", RMSE_rpart2)


#Building Random Forest Model
library(randomForest)
randomForestmodel2 <- randomForest(taste ~ . - quality, data = train, ntree= 500, mtry=6)
predictrandomForestmodel2 <- predict(randomForestmodel2, data= test)
x<-table(test$taste, predictrandomForestmodel2)
accuracy2= (sum(diag(x)))/sum(x)
sprintf("Accuracy of Random Forest is %f", accuracy2)
library(caret)
# Precision - Fractions of correct predicitions for a class
precision = diag(x)/apply(x,2,sum) 
# Recall - Fractions of instances of a class that were correctly predicted
recall = diag(x)/apply(x,1,sum)
f1 = 2*precision*recall/(precision+recall)
data.frame(precision, recall, f1) 

#Building SVM Classification Model 
library(e1071)
svmmodel2<-svm(taste ~. -quality, data=train)
predictsvmmodel2 <- predict(svmmodel2 ,test)
x<-table(test$taste, predictsvmmodel2)
accuracy2= (sum(diag(x)))/sum(x)
sprintf("Accuracy of SVM is %f", accuracy2)
library(caret)
# Precision - Fractions of correct predicitions for a class
precision = diag(x)/apply(x,2,sum) 
# Recall - Fractions of instances of a class that were correctly predicted
recall = diag(x)/apply(x,1,sum)
f1 = 2*precision*recall/(precision+recall)
data.frame(precision, recall, f1) 


# Finding correlations between predictors and quality
library(corrplot)
corrplot(cor(as.matrix(wine[,1:11])), method= "number")
#Automatic Feature Selection 
#Feature Selection with the Caret R Package
library(caret)
#ctrl <- rfeControl(method = "repeatedcv",repeats = 5, verbose = TRUE, functions = rfFuncs)
#wineRFE <- rfe(x = wine[,1:11], y = wine[,13], sizes = c(1:11), metric = "Accuracy", rfeControl = ctrl)
#plot(wineRFE, type = c("g", "o"))

#Exploring Principal Components 
require(ggfortify)
library(ggfortify)
Wineclasses<-factor(wine$taste)
autoplot(prcomp(wine[,1:11]), data= wine, label =FALSE, loadings=TRUE, loadings.label =TRUE, colour= 'taste')
