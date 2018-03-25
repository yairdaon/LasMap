#!/usr/bin/Rscript
library( "rEDM" )
data <- read.csv("tests/data/2NN.csv",
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

data <- read.csv("tests/data/3NN.csv",
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


data <- read.csv("tests/data/generic_sets.csv",
                 header = TRUE)

pred <- block_lnlp(data,
                   lib=c(1,5),
                   pred=c(6,10),
                   norm_type="L2 norm",
                   P = 0.5,
                   method="simplex",
                   tp=0,
                   num_neighbors="E+1",
                   columns=c("V1", "V2", "V3"),
                   target_column = "target",
                   stats_only=FALSE )$model_output[[1]]
pred <- pred[ , c("obs", "pred") ]
print( pred )

## Test simplex.univariate
py <- read.csv("tests/data/univariate.csv",
               header = TRUE,
               na.strings=c("NaN"))

output <- simplex(py[ , c("time", "truth") ],
                  E=2,
                  tp=2,
                  stats_only=FALSE)$model_output[[1]]

output <- output[ , c("time", "pred","obs") ]

ret <- merge(x=py,
             y=output,
             by="time",
             all=TRUE,
             suffixes=c("-Python", "-R") )
print(ret)
