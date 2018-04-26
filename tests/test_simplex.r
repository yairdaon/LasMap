library( "rEDM" )

print( "Comparing 2 nearest neighbours, R vs Python.")
data <- read.csv("tests/data/simple_data.csv",
                 header = TRUE)
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
R <- out_2NN[ , "pred"]
py <- read.table("tests/data/2NN.txt")[ , "V1" ]
stopifnot( all.equal(R, py, tolerance=1e-10 ) ) ## Ensure 2NN results agree


print( "Comparing 3 nearest neighbours, R vs Python.")
data <- read.csv("tests/data/simple_data.csv",
                 header = TRUE)
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
R <- out_3NN[ , "pred"]
py <- read.table("tests/data/3NN.txt")[ , "V1" ]
stopifnot( all.equal(R, py, tolerance=1e-10 ) ) ## Ensure 3NN results agree


print( "Comparing set predictions, R vs Python.")
data <- read.csv("tests/data/generic_sets.csv",
                 header = TRUE)
out_sets <- block_lnlp(data,
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
stopifnot( all.equal(out_sets[ , "obs"],
                     out_sets[ , "pred"],
                     tolerance=1e-10 ) ) ## Ensure sets results agree

print( "Comparing univariate prediction, R vs Python.")
py <- read.csv("tests/data/univariate.csv",
               header = TRUE,
               na.strings=c("NaN"))

R <- simplex(py[ , c("time", "truth") ],
             E=2,
             tp=2,
             stats_only=FALSE)$model_output[[1]]
R <- R[ , c("time", "pred","obs") ]

ret <- merge(x=py,
             y=R,
             by="time",
             all=FALSE,
             suffixes=c("-Python", "-R") )

comparison <- all.equal(ret[ , "pred-Python"],
                        ret[ , "pred-R"],
                        tolerance=1e-10 )
print( comparison )
print( ret )
## if( !comparison )
## {
##     print( ret )
##     stop("Test failed, see above printout.")
## }
