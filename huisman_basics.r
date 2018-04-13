library( "rEDM" )

df <- read.csv("huisman/noisy_truncated_huisman.csv",
               header = TRUE,
               sep = ",",
               na.strings = c( "NA", "NaN" ),
               row.names=1)

## Add rows of sums
df["N3+N5"] <- df["N3"] + df["N5"]
df["N2+N4"] <- df["N2"] + df["N4"]

## normalize
df <- as.data.frame(scale(df))

for( variable in names( df ) )
{
    print( variable )
    output <- simplex(df[,variable])
    png(paste0("huisman/pix/R/", variable, "_skill_full.png"))
    plot(output$E, output$rho, type = "l",
         xlab = "Embedding Dimension (E)",
         ylab = "Forecast Skill (rho)")
    dev.off()
}
