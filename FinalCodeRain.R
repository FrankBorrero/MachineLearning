library(corrplot)
library(ppcor)
library(dplyr)
library(GGally)
library(tseries)
library(purrr)
library(tidyr)
library(readxl)
library(recipes)
library(mlr)
library(mlbench)
library(e1071)
library(kknn)
library(rpart)
library(kernlab)
library(nnet)
library(unbalanced)
library(DiscriMiner)
library(FSelectorRcpp)
library(praznik)
library(randomForest)
library(ada)
library(RWeka)
## GETTING INTO DATASET
#We load the dataset
DatasetAus <- read.csv(file="D:/Maestria UPM/M6 Machine learning/HW/dataset/weatherAUS.csv", header=TRUE, sep=",")
#Preview
View(DatasetAus)
#List the variables
names(DatasetAus)
names(DatasetAus) <- tolower(names(DatasetAus))




# list the structure of DataWeather, look for NA
str(DatasetAus)

# Dimension from DataWeather 
dim(DatasetAus)
# print first 10 rows of DataWeather
head(DatasetAus, n=10)
#Summarize the dataset
summary(DatasetAus)


#install.packages("naniar") 
library(naniar)
gg_miss_var(DatasetAus) # In the plot we can evidence that sushine, evaporation, cloud3pm, cloud9am has  too much missing values, so we proceed to delete them
library(dplyr)
#install.packages("ggplot2")
library(ggplot2)
#Barplot to  evidence proportion from  output variable raintomorrow
ggplot(DatasetAus, aes(raintomorrow)) +
  geom_bar(fill = "#0073C2FF") 
tableperc=table(DatasetAus$raintomorrow)
View(tableperc)
print(tableperc)

###   PREPARING THE DATASET


#We Separate Year-Month-DAy from date as  we we know the month alone plays a significant role for the prediction of Rainy days.
DatasetAus<-DatasetAus %>%dplyr::mutate(year = lubridate::year(date), 
                                        month = lubridate::month(date), 
                                        day = lubridate::day(date))
##Dropped the risk_mm as said in dataset explanation
#Dropped columns sunshine,evaporation,cloud3pm,cloud9am  because lot of missed values
#Aldo dropped categorical variables  and date column

DatasetAus <- subset(DatasetAus, select = -c(risk_mm, location,windgustdir,winddir9am,winddir3pm,date))
str(DatasetAus)
head(DatasetAus, n=10)

library(tidyr)
sum(is.na(DatasetAus))
DatasetAusNA<- DatasetAus%>% drop_na()

#Validating non NA exist
gg_miss_var(DatasetAusNA) # In the plot we can evidence that non na exists

#Percentage of raintomorrow
ggplot(DatasetAusNA, aes(raintomorrow)) +
  geom_bar(fill = "#0073C2FF") 
tablepercAUS=table(DatasetAusNA$raintomorrow)
View(tablepercAUS)
tablepercAUS
#Convert raintoday data to numeric
library(plyr)
DatasetAusNA$raintoday <- revalue(DatasetAusNA$raintoday, c("Yes"=1))
DatasetAusNA$raintoday <- revalue(DatasetAusNA$raintoday, c("No"=0))
str(DatasetAusNA)
DatasetAusNA$raintoday=as.factor(DatasetAusNA$raintoday)
str(DatasetAusNA)

#Convert Raintomrrow data to numeric
library(plyr)
DatasetAusNA$raintomorrow <- revalue(DatasetAusNA$raintomorrow, c("Yes"=1))
DatasetAusNA$raintomorrow <- revalue(DatasetAusNA$raintomorrow, c("No"=0))
str(DatasetAusNA)
DatasetAusNA$raintomorrow=as.factor(DatasetAusNA$raintomorrow)
str(DatasetAusNA)



#correlation matrix compute
#install.packages("varhandle") 
library(varhandle) 
#DataWeather3 <- subset(DataWeather3, select = -c(location))
DatasetAusNA$raintoday<- unfactor(DatasetAusNA$raintoday)
DatasetAusNA$raintomorrow<- unfactor(DatasetAusNA$raintomorrow)
cormat <- round(cor(DatasetAusNA), 1)
head(cormat)



#install.packages("ggcorrplot")
library(ggcorrplot)
ggcorrplot(cormat)

# Reordering the correlation matrix
# --------------------------------
# using hierarchical clustering

ggcorrplot(cormat, hc.order = TRUE, outline.col = "white",insig = "blank")
#Variables related to pressure and temperature are highly correlated. 
#I generated some boxplots and found that no single variable is 
#better than another for differentiating between rain/no rain, so I'll 
#just use the temperature and pressure at 3pm and remove the other 
#predictors.

### NORMALIZATION #Standardizing the data using min-max normalization
##the normalization function is created
normal <-function(x) { (x -min(x))/(max(x)-min(x))   }
str(DatasetAusNA)
DatasetAusNor <- as.data.frame(lapply(DatasetAusNA[], normal))
View(DatasetAusNor)

###PROBANDO MLR
ListLearners=listLearners()
#install.packages("mlr")
library(mlr)

DatasetAusNor$raintoday<- as.factor(DatasetAusNor$raintoday)
DatasetAusNor$raintomorrow<- as.factor(DatasetAusNor$raintomorrow)
str(DatasetAusNor)
library(caTools)
#SPLIT
set.seed(123)
split = sample.split(DatasetAusNor$raintomorrow, SplitRatio = 0.70)
split = sample.split(DatasetAusNor, SplitRatio = 0.70)
AusTraining_set = subset(DatasetAusNor, split == TRUE)
AusTest_set = subset(DatasetAusNor, split == FALSE)
AusTraining_set$raintoday<- unfactor(AusTraining_set$raintoday)
AusTraining_set$raintomorrow<- factor(AusTraining_set$raintomorrow)
AusTest_set$raintoday<- unfactor(AusTest_set$raintoday)
AusTest_set$raintomorrow<- factor(AusTest_set$raintomorrow)

str(AusTraining_set)
str(AusTest_set)


#Generate task
taskAusTrain = makeClassifTask(id = "DataAusTrainTask", data = AusTraining_set,
                               target = "raintomorrow", positive = "1")
gg_miss_var(AusTraining_set) # In the plot we can evidence that non na exists
gg_miss_var(AusTest_set) # In the plot we can evidence that non na exists
taskAusTrain
undersample(taskAusTrain, 0.28, cl = NULL)
taskAusTrainUnder=undersample(taskAusTrain, 0.28, cl = NULL)
taskAusTrainUnder
taskAusTest = makeClassifTask(id = "DataAusTestTask", data = AusTest_set,
                              target = "raintomorrow", positive = "1")
taskAusTest
taskAusTestUnder=undersample(taskAusTest, 0.3, cl = NULL)
taskAusTestUnder

str(getTaskData(taskAusTrainUnder))
getTaskDesc(taskAusTrainUnder)
str(getTaskData(taskAusTestUnder))
getTaskDesc(taskAusTestUnder)
## KNN
##Generate Learner KNN
lrn.knn <- makeLearner("classif.knn", k=140)
print(lrn.knn)
getParamSet(lrn.knn)


#train model
#taskAusTrainUnder <- createDummyFeatures(taskAusTrainUnder)
str(AusTraining_set)
str(AusTest_set)
knnmodel <- train(lrn.knn, task = taskAusTrainUnder)

names(knnmodel)
knnmodel$learner
knnmodel$features
knnmodel$subset
print(knnmodel)
getLearnerModel(knnmodel)

#predict on test data
knnpredict <- predict(knnmodel, taskAusTestUnder,na.action = na.pass)
View(knnpredict)
head(as.data.frame(knnpredict), 10)
pred.class = getPredictionResponse(knnpredict) # predicted classes
View(pred.class)
truthknn = getPredictionTruth(knnpredict)
head(truthknn, 3)
sum(pred.class != truthknn) # total number of errors
mean(pred.class == truthknn) # percentage of accurate predictions (ACC, accuracy)
# Making the Confusion Matrix
calculateConfusionMatrix(knnpredict,relative = TRUE, sums = TRUE)
#Evaluating
performance(knnpredict, measures = list(mlr::acc,  mlr::kappa))

plotLearnerPrediction(lrn.knn, taskAusTestUnder)

plotLearnerPrediction(lrn.knn, taskAusTrainUnder)
##need propabilistci
d = generateThreshVsPerfData(knnpredict, measures = list(acc, kappa, mmce))
plotThreshVsPerf(d)                      
##

##FILTER METHODS 
#Feature importance
#install.packages("FSelector")
library(FSelector)
filteredTask <- generateFilterValuesData(taskAusTrainUnder, method = c("FSelector_chi.squared"),abs=2)
filteredTask$data
plotFilterValues(filteredTask,n.show = 20) 
library(mlr)
plotFilterValuesGGVIS(im_feat)



##FSS-Filter KNN
lrnknnfilter <- makeFilterWrapper(learner = "classif.knn", 
                                  fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6, )
rdesc = makeResampleDesc("CV", iters = 10)
knnresamplefilter <- resample(learner =lrnknnfilter, task =taskAusTrainUnder, resampling = rdesc, show.info = FALSE, models = TRUE, measures = list (mlr::acc, mlr::mmce))
knnresamplefilter$aggr

modelknnfss <- mlr::train(lrnknnfilter, taskAusTrainUnder)
predictknnfss <- predict(modelknnfss, taskAusTestUnder )
performance(predictknnfss, measures = list(acc, mmce, kappa))

#Wrapper 
lrnknnwra <- makeFeatSelWrapper(learner = "classif.kknn",
                                  resampling = rdesc, control = 
                                    makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
resampleknnwra <- resample(lrnknnwra, taskAusTrain, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
resampleknnwra$aggr
modelknnwra <- mlr::train(lrnknnwra, taskAusTrainUnder)
predictknnwra <- predict(modelknnwra, taskAusTrainUnder)
predictknnwra
performance(predictknnwra, measures = list(acc, mmce, kappa))

#####DECISION TREE
##generate learner
getParamSet("classif.rpart")
lrn.dt <- makeLearner("classif.rpart", 
                      predict.type = "response")
print(lrn.dt)
getParamSet(lrn.dt)

#Train
dtmodel <- train(lrn.dt, task=taskAusTrainUnder)
names(dtmodel)
print(dtmodel)
getLearnerModel(dtmodel)

#predict on test data
dtpredict <- predict(dtmodel, task = taskAusTestUnder)
head(as.data.frame(dtpredict))


##evaluating ##validate
truthdt = getPredictionTruth(dtpredict)
head(truthdt, 3)
sum(pred.class != truthdt) # total number of errors
mean(pred.class == truthdt) # percentage of accurate predictions (ACC, accuracy)
calculateConfusionMatrix(dtpredict)
performance(dtpredict, measures = list(mlr::acc, mlr::mmce, mlr::kappa))

plotLearnerPrediction(lrn.dt, taskAusTestUnder)

plotLearnerPrediction(lrn.dt, taskAusTrainUnder)

#FSS-Filter TREE
lrndtfilter <- makeFilterWrapper(learner = "classif.rpart", 
                                    fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.8)
dtresamplefilter <- resample(learner =lrndtfilter, task =taskAusTrainUnder, resampling = rdesc, show.info = FALSE, models = TRUE, measures = list (mlr::acc, mlr::mmce))
dtresamplefilter$aggr

dtmodelfss <- mlr::train(lrndtfilter, taskAusTrainUnder)
predictdtfs <- predict(dtmodelfss, taskAusTestUnder )
performance(predictdtfs, measures = list(acc, mmce))

#Wrapper 
lrndtwra <- makeFeatSelWrapper(learner = "classif.rpart",
                                    resampling = rdesc, control = 
                                      makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
dtresamplewra <- resample(lrndtwra, taskAusTrainUnder, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
dtresamplewra$aggr
moddtwra <- mlr::train(lrndtwra, taskAusTrainUnder)
predictdtwra <- predict(moddtwra, taskAusTestUnder)
performance(predictdtwra, measures = list(acc, mmce, kappa))


#Rule Induction 
##generate learner
getParamSet("classif.JRip")
lrn.ri <- makeLearner("classif.JRip", 
                      predict.type = "response")
print(lrn.ri)
getParamSet(lrn.ri)

#Train
rimodel <- train(lrn.ri, task=taskAusTrainUnder)
names(rimodel)
print(rimodel)
getLearnerModel(rimodel)

#predict on test data
ripredict <- predict(rimodel, task = taskAusTestUnder)
head(as.data.frame(ripredict))


##evaluating ##validate
truthri = getPredictionTruth(ripredict)
head(truthri, 3)
sum(pred.class != truthri) # total number of errors
mean(pred.class == truthri) # percentage of accurate predictions (ACC, accuracy)
calculateConfusionMatrix(ripredict)
performance(ripredict, measures = list(mlr::acc, mlr::mmce, mlr::kappa))

plotLearnerPrediction(lrn.ri, taskAusTestUnder)


#FSS-Filter rule induction
lrnrnfilter <- makeFilterWrapper(learner = "classif.JRip", 
                                     fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
riresamplefilter <- resample(learner =lrnrnfilter, task =taskAusTrainUnder, resampling = rdesc, show.info = FALSE, models = TRUE, measures = list (mlr::acc, mlr::mmce))
riresamplefilter$aggr

modelrifss <- mlr::train(lrnrnfilter, taskAusTrainUnder)
predictrifss <- predict(modelrifss, taskAusTestUnder )
performance(predictrifss, measures = list(acc, mmce))

#Wrapper 
lrnriwra <- makeFeatSelWrapper(learner = "classif.JRip",
                                     resampling = rdesc, control = 
                                       makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
riresamplewra <- resample(lrnriwra, taskAusTrainUnder, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
riresamplewra$aggr
modriwra <- mlr::train(lrnriwra, taskAusTrainUnder)
predictriwra <- predict(modriwra, taskAusTestUnder)
performance(predictriwra, measures = list(acc, mmce))


##########################
##SVM#####################
#Rule Induction 
##generate learner
getParamSet("classif.svm")
lrn.svm <- makeLearner("classif.svm", 
                       predict.type = "response")
print(lrn.svm)
getParamSet(lrn.svm)

#Train
svmmodel <- train(lrn.svm, task=taskAusTrainUnder)
names(svmmodel)
print(svmmodel)
getLearnerModel(svmmodel)

#predict on test data
svmpredict <- predict(svmmodel, task = taskAusTestUnder)
head(as.data.frame(svmpredict))


##evaluating ##validate
truthsvm = getPredictionTruth(svmpredict)
head(truthsvm, 3)
sum(pred.class != truthsvm) # total number of errors
mean(pred.class == truthsvm) # percentage of accurate predictions (ACC, accuracy)
calculateConfusionMatrix(svmpredict)
performance(svmpredict, measures = list(mlr::acc, mlr::mmce, mlr::kappa))

plotLearnerPrediction(lrn.svm, taskAusTestUnder)


#FSS-Filter SVN
lrnsvmfilter <- makeFilterWrapper(learner = "classif.ksvm", 
                                      fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
rdesc.svm <- makeResampleDesc("CV")
svmresamplefilter <- resample(learner =lrnsvmfilter, task =taskAusTrainUnder, resampling = rdesc.svm, show.info = FALSE, models = TRUE, measures = list (mlr::acc, mlr::mmce))
svmresamplefilter$aggr

modsvmfilter <- mlr::train(lrnsvmfilter, taskAusTrainUnder)
predictsvmfss <- predict(modsvmfilter, taskAusTestUnder )
performance(predictsvmfss, measures = list(acc, mmce))

#Wrapper 
lrnsvmwra <- makeFeatSelWrapper(learner = "classif.ksvm",
                                      resampling = rdesc, control = 
                                        makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
svmresamplewra <- resample(lrnsvmwra, taskAusTrainUnder, resampling = rdesc.svm, models = TRUE, show.info = FALSE, measures = mlr::acc)

modsvmwra <- mlr::train(lrnsvmwra, taskAusTrainUnder)
predictsvmwra <- predict(modsvmwra, taskAusTestUnder)
performance(predictsvmwra, measures = list(acc, mmce, kappa))


##########################
##ARTIFICIAL NEURAL NETWORK#####################
##generate learner
getParamSet("classif.nnet")
lrn.ann <- makeLearner("classif.nnet", 
                       predict.type = "response")
print(lrn.ann)
getParamSet(lrn.ann)

#Train
annmodel <- train(lrn.ann, task=taskAusTrainUnder)
names(annmodel)
print(annmodel)
getLearnerModel(annmodel)

#predict on test data
annpredict <- predict(annmodel, task = taskAusTestUnder)
head(as.data.frame(annpredict))


##evaluating ##validate
truthann = getPredictionTruth(annpredict)
head(truthann, 3)
sum(pred.class != truthann) # total number of errors
mean(pred.class == truthann) # percentage of accurate predictions (ACC, accuracy)
calculateConfusionMatrix(annpredict)
performance(annpredict, measures = list(mlr::acc, mlr::mmce, mlr::kappa))

plotLearnerPrediction(lrn.ann, taskAusTestUnder)


#FSS-Filter neural network
lrnannfilter <- makeFilterWrapper(learner = "classif.nnet", 
                                     fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
annresamplefilter <- resample(learner =lrnannfilter, task =taskAusTrainUnder, resampling = rdesc, show.info = FALSE, models = TRUE, measures = list (mlr::acc, mlr::mmce))
annresamplefilter$aggr

modannfilter <- mlr::train(lrnannfilter, taskAusTrainUnder)
predictannfss <- predict(modannfilter, taskAusTestUnder )
performance(predictannfss, measures = list(acc, mmce))

#Wrapper 
lrnannwra <- makeFeatSelWrapper(learner = "classif.nnet",
                                     resampling = rdesc, control = 
                                       makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
annresamplewra <- resample(lrnannwra, taskAusTrainUnder, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
annresamplewra$aggr
modannwra <- mlr::train(lrnannwra, taskAusTrainUnder)
predictannwra <- predict(modannwra, taskAusTestUnder)
performance(predictannwra, measures = list(acc, mmce))

##### PROBABILISTICS ########################
##########################
##LOGISTIC REGRESSION#####################
##generate learner
getParamSet("classif.logreg")
lrn.lr <- makeLearner("classif.logreg", 
                      predict.type = "response")
print(lrn.lr)
getParamSet(lrn.lr)

#Train
lrmodel <- train(lrn.lr, task=taskAusTrainUnder)
names(lrmodel)
print(lrmodel)
getLearnerModel(lrmodel)

#predict on test data
lrpredict <- predict(lrmodel, task = taskAusTestUnder)
head(as.data.frame(lrpredict))


##evaluating ##validate
truthlr = getPredictionTruth(lrpredict)
head(truthlr, 3)
sum(pred.class != truthlr) # total number of errors
mean(pred.class == truthlr) # percentage of accurate predictions (ACC, accuracy)
calculateConfusionMatrix(lrpredict)
performance(lrpredict, measures = list(mlr::acc, mlr::mmce, mlr::kappa))

plotLearnerPrediction(lrn.lr, taskAusTestUnder)

#FSS-Filter LOG
lrnlrfilter <- makeFilterWrapper(learner = "classif.logreg", 
                                    fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
lrresamplefilter <- resample(learner =lrnlrfilter, task =taskAusTrainUnder, resampling = rdesc, show.info = FALSE, models = TRUE, measures = list (mlr::acc, mlr::mmce))
lrresamplefilter$aggr

modlrfil <- mlr::train(lrnlrfilter, taskAusTrainUnder)
predictlgfi <- predict(modlrfil,  taskAusTestUnder )
performance(predictlgfi, measures = list(acc, mmce))

#Wrapper 
lrnlrwra <- makeFeatSelWrapper(learner = "classif.logreg",
                                    resampling = rdesc, control = 
                                      makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
lgresamplewra <- resample(lrnlrwra, taskAusTrainUnder, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
lgresamplewra$aggr
modellrwra <- mlr::train(lrnlrwra, taskAusTrainUnder)
predictlrwra <- predict(modellrwra, taskAusTestUnder)
performance(predictlrwra, measures = list(acc, mmce, kappa))




##Naive Bayes#####################
##generate learner
getParamSet("classif.naiveBayes")
lrn.nb <- makeLearner("classif.naiveBayes", 
                      predict.type = "response")
print(lrn.nb)
getParamSet(lrn.nb)

#Train
nbmodel <- train(lrn.nb, task=taskAusTrainUnder)
names(nbmodel)
print(nbmodel)
getLearnerModel(nbmodel)

#predict on test data
nbpredict <- predict(nbmodel, task = taskAusTestUnder)
head(as.data.frame(nbpredict))


##evaluating ##validate
truthnb = getPredictionTruth(nbpredict)
head(truthnb, 3)
sum(pred.class != truthnb) # total number of errors
mean(pred.class == truthnb) # percentage of accurate predictions (ACC, accuracy)
calculateConfusionMatrix(nbpredict)
performance(nbpredict, measures = list(mlr::acc, mlr::mmce, mlr::kappa))
plotLearnerPrediction(lrn.nb, taskAusTestUnder)


#FSS-Filter NB
lrnnbfilter <- makeFilterWrapper(learner = "classif.naiveBayes", 
                                     fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
nbresamplefilter <- resample(learner =lrnnbfilter, task =taskAusTrainUnder, resampling = rdesc, show.info = FALSE, models = TRUE, measures = list (mlr::acc, mlr::mmce))
nbresamplefilter$aggr

modnbfss <- mlr::train(lrnnbfilter, taskAusTrainUnder)
predictnbfss <- predict(modnbfss, taskAusTestUnder )
performance(predictnbfss, measures = list(acc, mmce, kappa))

#Wrapper 
lrnnbwra <- makeFeatSelWrapper(learner = "classif.naiveBayes",
                                     resampling = rdesc, control = 
                                       makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
nbresamplewrap <- resample(lrnnbwra, taskAusTrainUnder, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
nbresamplewrap$aggr
modnbwra <- mlr::train(lrnnbwra, taskAusTrainUnder)
predictnbwra <- predict(modnbwra, taskAusTestUnder)
performance(predictnbwra, measures = list(acc, mmce, kappa))


##Discriminant analysis#####################
##generate learner
install.packages("DiscriMiner")
library("DiscriMiner")
getParamSet("classif.linDA")
lrn.da<- makeLearner("classif.linDA", 
                     predict.type = "response")
print(lrn.da)
getParamSet(lrn.da)

#Train
damodel <- train(lrn.da, task=taskAusTrainUnder)
names(damodel)
print(damodel)
getLearnerModel(damodel)

#predict on test data
dapredict <- predict(damodel, task = taskAusTestUnder)
head(as.data.frame(dapredict))


##evaluating ##validate
truthda = getPredictionTruth(dapredict)
head(truthda, 3)
sum(pred.class != truthda) # total number of errors
mean(pred.class == truthda) # percentage of accurate predictions (ACC, accuracy)
calculateConfusionMatrix(dapredict)
performance(dapredict, measures = list(mlr::acc, mlr::mmce, mlr::kappa))
plotLearnerPrediction(lrn.da, taskAusTestUnder)

#FSS-Filter LDiscriminat A
lrndafilter <- makeFilterWrapper(learner = "classif.linDA", 
                                     fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
daresamplefilter <- resample(learner =lrndafilter, task =taskAusTrainUnder, resampling = rdesc, show.info = FALSE, models = TRUE, measures = list (mlr::acc, mlr::mmce))
daresamplefilter$aggr

moddafss <- mlr::train(lrndafilter, taskAusTrainUnder)
predictdafss <- predict(moddafss, taskAusTestUnder )
performance(predictdafss, measures = list(acc, mmce))

#Wrapper 
lrndawra <- makeFeatSelWrapper(learner = "classif.linDA",
                                     resampling = rdesc, control = 
                                       makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
daresamplewra <- resample(lrndawra, taskAusTrainUnder, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
daresamplewra$aggr
moddawra <- mlr::train(lrndawra, taskAusTrainUnder)
predictdawra <- predict(moddawra, taskAusTestUnder )
performance(predictdawra, measures = list(acc, mmce))




-------------------------------------------------------------------
 


#FSS-Filter LDiscriminat A
lrnknnfilter.da <- makeFilterWrapper(learner = "classif.linDA", 
                                     fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
knnresamplefilter.da <- resample(learner =lrnknnfilter.da, task =taskAusTrain, resampling = rdesc, show.info = FALSE, models = TRUE, measures = list (mlr::acc, mlr::mmce))
knnresamplefilter.da$aggr

mod.dt.fss.da <- mlr::train(lrnknnfilter.da, taskAusTrain)
predict.dt.fss.da <- predict(mod.dt.fss.da, taskAusTest )
performance(predict.dt.fss.da, measures = list(acc, mmce, kappa))

#Wrapper 
lrn.wra.knn.da <- makeFeatSelWrapper(learner = "classif.linDA",
                                     resampling = rdesc, control = 
                                       makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
r.knn.wra.da <- resample(lrn.wra.knn.da, taskAusTrain, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
r.knn.wra.da$aggr
mod.knn.wra.da <- mlr::train(lrn.wra.knn.da, taskAusTrain)
predict.knn.wra.da <- predict(mod.knn.wra.da, taskAusTest)
performance(predict.knn.wra.da, measures = list(acc, mmce, kappa))


#RandomForest

#learner
getParamSet("classif.randomForest")
learner.randomf <- makeLearner("classif.randomForest", 
                               predict.type = "response", ntree=100)
learner.randomf$par.set

#Train
mod.randomf <- mlr::train(learner.randomf, taskAusTrain)
getLearnerModel(mod.randomf)

#Predict
predict.randomf <- predict(mod.randomf, taskAusTest)
head(as.data.frame(predict.randomf))
calculateConfusionMatrix(predict.randomf)

#Performance
performance(predict.randomf, measures = list(acc, mmce))

RCV.randomf <- repcv(learner.randomf, taskAusTrain, folds = 3, reps = 2, 
                     measures = list(acc, mmce), stratify = TRUE)
RCV.randomf$aggr

#AdaBoos

#learner
getParamSet("classif.ada")
learner.ada <- makeLearner("classif.ada", 
                           predict.type = "response")
learner.ada$par.set

#Train
mod.ada <- mlr::train(learner.ada, taskAusTrain)
getLearnerModel(mod.ada)

#Predict
predict.ada <- predict(mod.ada, task = taskAusTest)
head(as.data.frame(predict.ada))
calculateConfusionMatrix(predict.ada)

#Performance
performance(predict.ada, measures = list(acc, mmce, kappa))

RCV.ada <- repcv(learner.ada, taskAusTrain, folds = 3, reps = 2, 
                 measures = list(acc, mmce, kappa), stratify = TRUE)
RCV.ada$aggr


##BENCHMARK


##Benchmark----

lrns <- list(lrn.knn, lrn.dt, lrn.ri,lrn.svm,lrn.nb,lrn.lr,lrn.da,lrn.ann
             )
rdesc <- makeResampleDesc("RepCV", folds = 3, reps = 2) #Choose the resampling strategy
bmr <- benchmark(lrns, taskAusTestUnder, rdesc, measures = list(acc, mmce,kappa))
getBMRPerformances(bmr, as.df = TRUE)
getBMRAggrPerformances(bmr, as.df = TRUE)

lrnsfss <- list(lrnknnfilter,lrndtfilter, lrnrnfilter, lrnsvmfilter,lrnnbfilter, lrnlrfilter, lrndafilter,lrnannfilter)
benchfss <- benchmark(lrnsfss, taskAusTestUnder, rdesc, measures = list(acc, mmce, kappa))
getBMRAggrPerformances(benchfss, as.df = TRUE)

lrnswra <- list(lrnknnwra,lrndtwra, lrnriwra, lrnsvmwra,
                 lrnnbwra, lrnlrwra, lrndawra,lrnannwra)
benchwra <- benchmark(lrnswra, taskAusTestUnder, rdesc, measures = list(acc, mmce, kappa))
getBMRAggrPerformances(benchwra, as.df = TRUE)



##Metaclassifiers 

#### AdaBoos

#Make learner
getParamSet("classif.ada")
lrnada <- makeLearner("classif.ada", 
                           predict.type = "response")
#Train
modelada <- mlr::train(lrnada, taskAusTrainUnder)
getLearnerModel(modelada)

#Predict
predictada <- predict(modelada, task = taskAusTestUnder)
head(as.data.frame(predictada))
calculateConfusionMatrix(predictada)
performance(predictada, measures = list(acc, mmce))

cvada <- repcv(lrnada, taskAusTrainUnder, folds = 3, reps = 2, 
                 measures = list(acc, mmce, kappa), stratify = TRUE)
cvada$ag



#RandomForest

#make learner
getParamSet("classif.randomForest")
lnrrf <- makeLearner("classif.randomForest", 
                               predict.type = "response", ntree=100)
lnrrf$par.set

#Train Model
modelrf <- mlr::train(lnrrf, taskAusTrainUnder)
getLearnerModel(modelrf)

#Predict
predictrf <- predict(modelrf, task = taskAusTestUnder)
calculateConfusionMatrix(predictrf)

#Performance
performance(predictrf, measures = list(acc, mmce))

cvrf <- repcv(lnrrf, taskAusTrainUnder, folds = 3, reps = 2, 
                     measures = list(acc, mmce), stratify = TRUE)
cvrf$aggr

##UNSUPERVISED


#RandomForest

install.packages("clue")
library(clue)
#learner
getParamSet("cluster.kmeans")
lnrkmeans <- makeLearner("cluster.kmeans", 
                               predict.type = "response")
lnrkmeans$par.set

#Train
taskAusTestClu = makeClusterTask(data = AusTraining_set)
AusTraining_setunfactor=AusTraining_set
AusTraining_setunfactor$raintomorrow<- unfactor(AusTraining_set$raintomorrow)

modkmeans <- mlr::train(lnrkmeans, taskAusTestClu)

getLearnerModel(mod.randomf)

#Predict
predict.randomf <- predict(mod.randomf, taskAusTest)
head(as.data.frame(predict.randomf))
calculateConfusionMatrix(predict.randomf)

#Performance
performance(predict.randomf, measures = list(acc, mmce))

RCV.randomf <- repcv(learner.randomf, taskAusTrain, folds = 3, reps = 2, 
                     measures = list(acc, mmce), stratify = TRUE)
RCV.randomf$aggr

