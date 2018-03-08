#!/usr/bin/Rscript
library(rEDM)
library(deSolve)
set.seed( 19 )

## clean directory from previous plots
system("rm -f huisman.pdf" )

gen_Huisman <- function(n, tau = 10)
{
    ## initial values & coefficients (Huisman & Weissing 2001, Fig. 2)
    N0 <- c(N1=0.1, N2=0.1, N3=0.1, N4=0.1, N5=0.1)
    R0 <- c(R1=10, R2=10, R3=10)
    
    num_N <- length(N0)
    num_R <- length(R0)
    
    
    ps <- list()
    ps$S <- R0
    ps$K <- matrix(c(0.20, 0.05, 0.50, 0.05, 0.50,
                     0.15, 0.06, 0.05, 0.50, 0.30,
                     0.15, 0.50, 0.30, 0.06, 0.05), nrow = num_R, byrow = TRUE)
    ps$C <- matrix(c(0.20, 0.10, 0.10, 0.10, 0.10,
                     0.10, 0.20, 0.10, 0.10, 0.20,
                     0.10, 0.10, 0.20, 0.20, 0.10), nrow = num_R, byrow = TRUE)
    ##   ps$K <- matrix(c(0.20, 0.05, 1.00, 0.05, 1.20, 
    ##                 0.25, 0.10, 0.05, 1.00, 0.40, 
    ##                 0.15, 0.95, 0.35, 0.10, 0.05), nrow = num_R, byrow = TRUE)
    ##  ps$C <- matrix(c(0.20, 0.10, 0.10, 0.10, 0.10, 
    ##                 0.10, 0.20, 0.10, 0.10, 0.20, 
    ##                 0.10, 0.10, 0.20, 0.20, 0.10), nrow = num_R, byrow = TRUE) ## step function
    ps$r <- rep.int(1, num_N) ## maximum growth rate
    ps$m <- rep.int(0.25, num_N) ## mortality
    ps$D <- 0.25
    

    
    dF <- function(t,x,ps)
    {
        N <- x[1:num_N]
        R <- x[num_N+(1:num_R)]
        mu <- ps$r * apply(R / (ps$K + R), 2, min)
        
        return(list(c(N * (mu - ps$m) ,
                      ps$D * (ps$S - R) - colSums(t(ps$C) * mu * N))))
    }

    timepoints <- seq(1,n)*tau
    model_data <- lsoda(c(N0,R0), timepoints, dF, parms = ps,hmax = 0.01, maxsteps = 5000)
    ## model_data <- ode(c(N0,R0),timepoints,dF,parms =  ps,hini = 0.01,"rk4")
    model_data <- as.data.frame(model_data)
    
    ## return(list(N = N, R = R))
    ## return(model_data[,1:(num_N+1)])
    return(model_data)
}


if( !file.exists("huisman.csv") ) {

    ## If no simulation data exists, generate it
    print( "Generating Huisman data" )
    huisman <- gen_Huisman(600)
    write.csv(huisman,
              "huisman.csv",
              quote = FALSE,
              row.names = FALSE)
} else {
    huisman <- read.csv("huisman.csv",
                        header=TRUE)
}
huisman$time <- NULL
huisman <- huisman[ 300:500, ]
print( names(huisman) )

## Normalize and add noise 
huisman <- data.frame(scale(huisman)) + runif(prod(dim(huisman))) * sqrt(0.1)

pdf("huisman.pdf"  )
old.par <- par(mfrow=c(2, 3))
cols <- c("N1", "N2", "N3", "N4", "N5")
for( col in cols )
{
    ts <- huisman[, col]
    print( ts[1:4] )
    ## Run simplex with CV
    output <- simplex(ts,
                      stats_only = TRUE,
                      silent = TRUE)
    
    ## Plot
    plot(output$E,
         output$rho,
         type = "l",
         xlab = "E",
         ylab = "rho",
         main = paste0("Univariate prediction for ", col ) )    
    grid(ny = NA,
         col = "red",
         lty = 1)
    ## print(paste0("Var ", col, " best E ", which(output$E == max(output$E)) ) ) 
}
par(old.par)
## plot(
dev.off()
