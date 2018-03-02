#!/usr/bin/Rscript
library( "rEDM" )
## source( "helpers/helper.r" )
## source( "helpers/plotting.r" )
## source( "helpers/mve.r" )

df <- read.csv("data/huisman.csv",
               header = TRUE,
               sep = ",",
               na.strings = c( "NA", "NaN" ),
               row.names=1)

## Add row of sum
df["sum"] <- df["N2"] + df["N4"]

## normalize
df <- as.data.frame(scale(df))
Es <- 1:9
rhos <- numeric( length(Es) )
for( variable in names( df ) )
{
    print( variable )
    output <- simplex(df[,variable])
    png(paste0("pix/", variable, "_skill_full_R.png"))
    plot(output$E, output$rho, type = "l",
         xlab = "Embedding Dimension (E)",
         ylab = "Forecast Skill (rho)")
    dev.off()
}