#!/usr/bin/Rscript
library( "rEDM" )
data <- read.csv("data/test_data_2NN.csv",
                 header = TRUE)

pred <- block_lnlp(data,
                   lib=c(1,4),
                   pred=c(5,5),
                   norm_type="L2 norm",
                   P = 0.5,
                   method="simplex",
                   tp=0,
                   num_neighbors=2,
                   columns=c("V1", "V2", "V3"),
                   target_column = "target",
                   stats_only=FALSE )$model_output[[1]]
pred <- pred[ , c("obs", "pred") ]
print( pred )

data <- read.csv("data/test_data_3NN.csv",
                 header = TRUE)

pred <- block_lnlp(data,
                   lib=c(1,4),
                   pred=c(5,5),
                   norm_type="L2 norm",
                   P = 0.5,
                   method="simplex",
                   tp=0,
                   num_neighbors=3,
                   columns=c("V1", "V2", "V3"),
                   target_column = "target",
                   stats_only=FALSE )$model_output[[1]]
pred <- pred[ , c("obs", "pred") ]
print( pred )


## df <- merge(x=data, y=pred, by="time", all.y=TRUE) 

## write.table(df,
##             file="data/test_results.csv",
##             quote=FALSE,
##             row.names=FALSE,
##             sep = ",")
