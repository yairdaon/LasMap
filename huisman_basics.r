library( "rEDM" )

df <- read.csv("Huisman/processed_huisman.csv",
               header = TRUE,
               sep = ",",
               na.strings = c( "NA", "NaN" ),
               row.names=1)

for( variable in names( df ) )
{
    print( variable )
    output <- simplex(df[,variable])
    png(paste0("Huisman/pix/R/", variable, "_skill_full.png"))
    plot(output$E, output$rho, type = "l",
         xlab = "Embedding Dimension (E)",
         ylab = "Forecast Skill (rho)")
    dev.off()
}
