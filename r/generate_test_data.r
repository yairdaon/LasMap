library( "rEDM" )
data <- read.csv("data/tests/simple_data.csv",
                 header = TRUE)

print( "Generating R comparison table for 2 nearest neighbours.")
out_2NN <- block_lnlp(data,
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
write.table(out_2NN[ , "pred"], "/data/tests/2NN.txt")[ , "V1" ]

print( "Generating R comparison table for 3 nearest neighbours.")
out_3NN <- block_lnlp(data,
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
write.table(out_3NN[ , "pred"], "data/tests/3NN.txt")[ , "V1" ]