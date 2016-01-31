library(readr)
library(xgboost)
library(ggplot2)
library(FeatureHashing)
library(dplyr)
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


names(train)[7:12] <- c("S1", "S2", "S3", "S4", "S5", "S6")
names(test)[7:12] <- c("S1", "S2", "S3", "S4", "S5", "S6")

train1 <- train[,c(2,3,7:12)]
test1 <- test[,c(2,3,7:12)]

feature_names <- names(train1)

#2-features count

for(i in 2:7){
  print("i")
  print(i)
  for(j in (i+1):8){
    print("j")
    print(j)
    feature_name <- feature_names[c(i,j)] 
    dots <- lapply(feature_name, as.symbol)  
    sel_data <- train1%>%
      group_by_(.dots=dots)   %>%
      select( i, j)  %>%
      summarise(count_i = n()/nrow(train1) ) 
    
    names(sel_data)[length(sel_data)] <- paste0("count",i,j) 
    
    
    
    train1 <- train1 %>% inner_join(sel_data,
                                     by=feature_name)

    test1 <- test1 %>% inner_join(sel_data,
                    by=feature_name)
    
    print("j end:")
    print(j)
  
  }
}


#3-features count now

train2 <- train1
test2 <- test1


for(i in 2:6){
  print("i")
  print(i)
  for(j in (i+1):7){
    print("j")
    print(j)
    for(k in (j+1):8 ){
      print("k")
      print(k)
    feature_name <- feature_names[c(i, j, k)] 
    dots <- lapply(feature_name, as.symbol)  
    sel_data <- train2%>%
      group_by_(.dots=dots)   %>%
      select( i, j, k)  %>%
      summarise(count_i = n()/nrow(train2) ) 
    
    names(sel_data)[length(sel_data)] <- paste0("count",i,j,k) 
    
    
    
    train2 <- train2 %>% inner_join(sel_data,
                                    by=feature_name)
    
    test2 <- test2 %>% inner_join(sel_data,
                                  by=feature_name)
    }   
  }
}

trainU <- data.frame(cbind(train2, train[,-c(2,3,7:12)]))
testU <- data.frame(cbind(test2, test[,-c(2,3,7:12)]))

######################################
#xgboost 
######################################

dtrain <- xgb.DMatrix(data=as.matrix(trainU), label=as.matrix(label[,1]))

watchlist=list(train=dtrain)

model_xgb_1 <- xgboost(data=dtrain,
                       max.depth=14,
                       eta=0.3,
                       gamma=3,
                       min_child_weight=5,
                       subsample=0.5,
                       colsample_bytree=0.5,
                       base_score=0,
                       nround=400,
                       nthread=6,
                       objective="multi:softmax",
                       num_class=3,
                       verbose=2,
                       watchlist=watchlist,
                       eval.metric="merror",
                       set.seed=13
)

pred <- data.frame(predict(model_xgb_1, as.matrix(testU)))
table(pred)
######################################
#End of xgboost model
######################################

#create sample submission
samplesub <- data.frame(cbind(samplesub, pred))
names(samplesub) <- c("ID", "Crop_Damage")
write.csv(samplesub, "submission/submission_31jan_3.csv", row.names=F)

#0.8439 on LB
