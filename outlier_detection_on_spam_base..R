#Outlier detection using Lof

#check the working directory.
getwd()
#import csvfile.
spambase <- read.csv("spambase.csv",header = TRUE,sep = ",")
#Finding the type of class.
class(spambase)
#Force the class variable to be nominal type
spambase$class = factor(spambase$class)
#select non spam emails.
spambaseNONSPAM = spambase[spambase$class == 0,]
#Remove class coloumn
spambaseNONSPAM = spambaseNONSPAM[,-58]
dim(spambaseNONSPAM)
spambaseNONSPAM
#local outlier factor (lof)
library(DDoutlier)
outlierness = LOF(dataset=spambaseNONSPAM, k=5)
names(outlierness) <- 1:nrow(spambaseNONSPAM)
sort(outlierness, decreasing = TRUE)
hist(outlierness)
which(outlierness > 2.0)


#Outlier detection using isolation forest.

#check the working directory.
getwd()
#import csv file
spambase <- read.csv("spambase.csv",header = TRUE,sep = ",")
#Finding the type of class.
class(spambase)
#Force the class variable to be nominal type
spambase$class = factor(spambase$class)
#select non spam emails.
spambaseNONSPAM = spambase[spambase$class == 0,]
#Remove class coloumn
spambaseNONSPAM = spambaseNONSPAM[,-58]
dim(spambaseNONSPAM)
spambaseNONSPAM
#Isolation forest
library(solitude)
#census data for spambase.
#data("spambaseNONSPAM",package = "mlbench")
#data("BostonHousing", pacakage = "mlbench")
iso <- isolationForest$new()
iso$fit(spambaseNONSPAM)
p <- iso$predict(spambaseNONSPAM)
print(p)
sort(p$anomaly_score)
plot(density(p$anomaly_score))
which(p$anomaly_score > 0.63)


#outlier detection using ocvsm.
#check the working directory.
getwd()
#import csvfile.
spambase <- read.csv("spambase.csv",header = TRUE,sep = ",")
#Finding the type of class.
class(spambase)
#Force the class variable to be nominal type
spambase$class = factor(spambase$class)
#select non spam emails.
spambaseNONSPAM = spambase[spambase$class == 0,]
#Remove class coloumn
spambaseNONSPAM = spambaseNONSPAM[,-58]
dim(spambaseNONSPAM)
spambaseNONSPAM
#data(spambaseNONSPAM)
df <- spambaseNONSPAM
model <- svm(df, y=NULL,type='one-classification')
print(model)
summary(model)

pred <- predict(model,df)
which(pred==TRUE)


