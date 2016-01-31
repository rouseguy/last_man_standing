#Benchmark code for last-man-standing hackathon by AnalyticsVidhya
#Code created on 30th Jan 2015
#Author: Bargava
#Link to competition
#http://datahack.analyticsvidhya.com/contest/last-man-standing

#This code does a couple of feature creation techniques
#1. two-way combination of columns
#2. Feature hashing

library(readr)
library(xgboost)
library(ggplot2)
library(FeatureHashing)
#Set the working directory. 

#Read the train, test and sample submission datasets
#The train datasets are the final training/testing dataset from the benchmark code
#Uses median imputation for missing values 
train <- read_csv("data/trainUpdated.csv")
test <- read_csv("data/testUpdated.csv")
label <- read_csv("data/labelsUpdated.csv")



#samplesub's first column is test's ID column
#Read the test dataset to get the ID column. 
#Needed to create the submission file
test1 <- read_csv("data/Test_C1XBIYq.csv")
samplesub <- as.matrix(test1$ID)
names(samplesub) <- c("ID")
rm(test1)


######################################
#Features to be added
######################################

#2-way combinations - concatenated
train1 <- c()

for(i in 1:ncol(train)){
  for(j in i:ncol(train)){
    train1 <- cbind(train1, 
                    as.integer(paste(train[,i],train[,j], sep="")))
  }
}

#2-way combinations - added
train2 <- c()

for(i in 1:ncol(train)){
  for(j in i:ncol(train)){
    train2 <- cbind(train1, train[,i] + train[,j])
  }
}

#2-way combinations - multiplied
train3 <- c()

for(i in 1:ncol(train)){
  for(j in i:ncol(train)){
    train3 <- cbind(train1, train[,i] * train[,j])
  }
}

#binarizing train1

train4 <- apply(train, 1, sum)
train4 <- data.frame(train4)
train5 <- apply(train4, 1, function(x) paste(rev((intToBits(x))), collapse=""))
train5 <- data.frame(as.character(train5))
train5[,1] <- as.character(train5[,1])
lapply(train5, class)
train6 <- c()
for(i in 1:64){
  train6 <- cbind(train6, apply(train5, 2, function(x) substr(x, i, i)))
}
for(i in 1:64){
  train6[,i] <- as.integer(train6[,i])
}
train6 <- data.frame(train6)
train6 <- sapply(train6, as.integer)
train6_headers <- colSums(train6)>nrow(train6)
train7 <- train6[,train6_headers]

#Create training dataset
train_set <- data.frame(cbind(train, 
                              train1, 
                              train2,
                              train3,
                              train7))

train_set <- sapply(train_set, as.numeric)

#feature creation for test

#2-way combinations - concatenated
test1 <- c()

for(i in 1:ncol(test)){
  for(j in i:ncol(test)){
    test1 <- cbind(test1, 
                   as.integer(paste(test[,i],test[,j], sep="")))
  }
}

#2-way combinations - added
test2 <- c()

for(i in 1:ncol(test)){
  for(j in i:ncol(test)){
    test2 <- cbind(test1, test[,i] + test[,j])
  }
}

#2-way combinations - multiplied
test3 <- c()

for(i in 1:ncol(test)){
  for(j in i:ncol(test)){
    test3 <- cbind(test1, test[,i] * test[,j])
  }
}

#binarizing test1

test4 <- apply(test, 1, sum)
test4 <- data.frame(test4)
test5 <- apply(test4, 1, function(x) paste(rev((intToBits(x))), collapse=""))
test5 <- data.frame(as.character(test5))
test5[,1] <- as.character(test5[,1])
lapply(test5, class)
test6 <- c()
for(i in 1:64){
  test6 <- cbind(test6, apply(test5, 2, function(x) substr(x, i, i)))
}
for(i in 1:64){
  test6[,i] <- as.integer(test6[,i])
}
test6 <- data.frame(test6)
test6 <- sapply(test6, as.integer)
test7 <- test6[,train6_headers]

#Create testing dataset
test_set <- data.frame(cbind(test, 
                             test1, 
                             test2,
                             test3,
                             test7))

test_set <- sapply(test_set, as.numeric)

######################################
#xgboost 
######################################

dtrain <- xgb.DMatrix(data=as.matrix(train_set), label=as.matrix(label[,1]))

watchlist=list(train=dtrain)

model_xgb_1 <- xgboost(data=dtrain,
                    max.depth=14,
                    eta=0.3,
                    gamma=3,
                    min_child_weight=5,
                    subsample=0.5,
                    colsample_bytree=0.5,
                    base_score=0,
                    nround=1200,
                    nthread=6,
                    objective="multi:softmax",
                    num_class=3,
                    verbose=2,
                    watchlist=watchlist,
                    eval.metric="merror",
                    set.seed=13
)

pred <- data.frame(predict(model_xgb_1, as.matrix(test_set)))
table(pred)
######################################
#End of xgboost model
######################################

#create sample submission
samplesub <- data.frame(cbind(samplesub, pred))
names(samplesub) <- c("ID", "Crop_Damage")
write.csv(samplesub, "submission/submission_30jan_1.csv", row.names=F)


#Scored 0.82 on the public LB

######################################
# Another Feature hashing
######################################

m.train <- hashed.model.matrix(~., train, 2^16)
m.test <- hashed.model.matrix(~., test, 2^16)

dtrain <- xgb.DMatrix(data=m.train, label=as.matrix(label[,1]))

watchlist=list(train=dtrain)

model_xgb_1 <- xgboost(data=dtrain,
                       max.depth=30,
                       eta=0.3,
                       gamma=3,
                       min_child_weight=5,
                       subsample=0.5,
                       colsample_bytree=0.5,
                       base_score=0,
                       nround=250,
                       nthread=6,
                       objective="multi:softmax",
                       num_class=3,
                       verbose=2,
                       watchlist=watchlist,
                       eval.metric="merror",
                       set.seed=13
)

pred <- data.frame(predict(model_xgb_1, m.test))
table(pred)
######################################
#End of xgboost model
######################################

#create sample submission
samplesub <- data.frame(cbind(samplesub, pred))
names(samplesub) <- c("ID", "Crop_Damage")
write.csv(samplesub, "submission/submission_30jan_2.csv", row.names=F)

# score on public LB: 0.82