## knitr::opts_chunk$set(echo = TRUE)
library('deSolve')
library('dplyr')
library('ggplot2')

set.seed(46892013)

params_original <- list(d = 0.11, m1 = 0.71, m2 = 0.43, a1 = 1, a2 = .85, R2 = 5)
# params <- list(dA = 0.11, dB = 0.12, mCynB = 0.71, mDinoA = 0.43, mDinoB = 0.41, 
#                 aCynB = 1, aDinoA = .85, aDinoB = .82, RCynB = .8, RDinoA = 5, RDinoB = 4.8,
#                 mld_critA = 22, mld_critB = 19)
# 
params <- list(dA = 0.11, dB = 0.14, mCynB = 0.71, mDinoA = 0.43, mDinoB = 0.38, 
                aCynB = 1, aDinoA = .85, aDinoB = .82, RCynB = .8, RDinoA = 5, RDinoB = 5.5,
                mld_critA = 10, mld_critB = 19)


state_vars <- c('NAdeep','NAeuph','NBdeep','NBeuph','CynB','DinoA','DinoB','QNA','QNB')

En_freq <- 15
EA_phases <-  runif(En_freq,0,2)
EA_amps <- seq(from=.5,to=1.5,length.out=En_freq)
EA_Ts <- seq(125,270,length.out = En_freq)*2

fEA <- function(ts) vapply(ts,function(t) {
    50 - sum( sapply(1:En_freq, function(i) 50*EA_amps[i]*sinpi(t/(EA_Ts[i])+EA_phases[i])^4/sum(EA_amps)) ) }
   ,1)

EB_phases <-  runif(En_freq,0,2)
EB_amps <- seq(from=.5,to=1.5,length.out=En_freq)
EB_Ts <- seq(100,250,length.out = En_freq)*2
fEB <- function(ts) vapply(ts,function(t) {50 - sum( sapply(1:En_freq, function(i) 50*EB_amps[i]*sinpi(t/(EB_Ts[i])+EB_phases[i])^4/sum(EB_amps)) )
},1)
```

Next are the nutrient fluxes.

```{r}
n_freq <- 10

QA_phases <-  runif(n_freq,0,2)
QA_amps <- seq(from=.75,to=1.25,length.out=n_freq)
QA_Ts <- seq(55,120,length.out = n_freq)*2

fQA <- function(t) 10*(sum( sapply(1:n_freq, function(i){
  QA_amps[i]*sinpi(t/QA_Ts[i]+QA_phases[i])^2/sum(QA_amps)
  })))^2


QB_phases <- runif(n_freq,0,2)
QB_amps <- seq(from=.75,to=1.5,length.out=n_freq)
QB_Ts <- seq(75,130,length.out = n_freq)*2
fQB <- function(t) 10*(sum( sapply(1:n_freq, function(i){
  QB_amps[i]*sinpi(t/QB_Ts[i]+QB_phases[i])^2/sum(QB_amps)
})))^2

# fQB <- function(t) 0
```

Now, we describe the dynamics when the mixed layer is deep, i.e. when both dinos and the CynB are utilizing the Neuph nutrient pools.

```{r,eval=FALSE}
### NORMAL
N_dNAdeep <- function(t,x,params){with( c(x,params), 
                      dA*(fQA(t) - NAdeep)
                                 )}
N_dNAeuph <- function(t,x,params){with( c(x,params),
                      dA*(NAdeep - NAeuph) - aCynB*NAeuph*CynB/(RCynB + NAeuph + NBeuph) - 
                        aDinoA*NAeuph*DinoA/(RDinoA + NAeuph)
                                )}
N_dNBdeep <- function(t,x,params){with( c(x,params), 
                      dB*(fQB(t) - NBdeep)
                                 )}
N_dNBeuph <- function(t,x,params){with( c(x,params),
                      dB*(NBdeep - NBeuph) - aCynB*NBeuph*CynB/(RCynB + NAeuph + NBeuph) - 
                        aDinoB*NBeuph*DinoB/(RDinoB + NBeuph)
                                )}

N_dDinoA <-  function(t,x,params){with( c(x,params), 
                                   aDinoA*NAeuph*DinoA/(RDinoA + NAeuph) - 
                                     (mDinoA)*DinoA + 0.01
)}
N_dDinoB <-  function(t,x,params){with( c(x,params), 
                                   aDinoB*NBeuph*DinoB/(RDinoB + NBeuph) - 
                                     (mDinoB)*DinoB + 0.01
)}
N_dCynB <-  function(t,x,params){with( c(x,params), 
                                   aCynB*(NAeuph + NBeuph)*CynB/(RCynB + NAeuph + NBeuph) - 
                                     (mCynB)*CynB + 0.01
)}
```

However, if mld (mixed layer depth) is less than mld_crit, the dinoflagellate species is able to take up nutrients from the Ndeep pool.


```{r}
### WITH THRESHOLD
dNAdeep <- function(t,x,params){with( c(x,params), 
                                      if(fEA(t) < mld_critA){
                                        return(dA*(fQA(t) - NAdeep) - aDinoA*NAdeep*DinoA/(RDinoA + NAdeep))
                                      }else{
                                        return(dA*(fQA(t) - NAdeep))
                                      }
)}
dNAeuph <- function(t,x,params){with( c(x,params),
                                      if(fEA(t) < mld_critA){
                                        return( dA*(NAdeep - NAeuph) - 
                                                  aCynB*NAeuph*CynB/(RCynB + NAeuph + NBeuph) )
                                      }else{
                                        return( dA*(NAdeep - NAeuph) - 
                                                  aCynB*NAeuph*CynB/(RCynB + NAeuph + NBeuph) - 
                                                  aDinoA*NAeuph*DinoA/(RDinoA + NAeuph) )
                                      }
)}


dNBdeep <- function(t,x,params){with( c(x,params), 
                                      if(fEB(t) < mld_critB){
                                        return(dB*(fQB(t) - NBdeep) - aDinoB*NBdeep*DinoB/(RDinoB + NBdeep))
                                      }else{
                                        return(dB*(fQB(t) - NBdeep))
                                      }
)}
dNBeuph <- function(t,x,params){with( c(x,params),
                                      if(fEB(t) < mld_critB){
                                        return( dB*(NBdeep - NBeuph) - 
                                                  aCynB*NBeuph*CynB/(RCynB + NAeuph + NBeuph) )
                                      }else{
                                        return( dB*(NBdeep - NBeuph) - 
                                                  aCynB*NBeuph*CynB/(RCynB + NAeuph + NBeuph) - 
                                                  aDinoB*NBeuph*DinoB/(RDinoB + NBeuph) )
                                      }
)}
                                      


dDinoA <-  function(t,x,params){with( c(x,params), 
                                      if(fEB(t) < mld_critA){
                                        return(aDinoA*NAdeep*DinoA/(RDinoA + NAdeep) - 
                                                 (mDinoA)*DinoA + 0.01)
                                      }else{
                                        return( aDinoA*NAeuph*DinoA/(RDinoA + NAeuph) - 
                                                  (mDinoA)*DinoA + 0.01)
                                        
                                      }
)}
dDinoB <- function(t,x,params){with( c(x,params), 
                                     if(fEA(t) < mld_critB){
                                       return(aDinoB*NBdeep*DinoB/(RDinoB + NBdeep) - 
                                                (mDinoB)*DinoB + 0.01)
                                     }else{
                                       return( aDinoB*NBeuph*DinoB/(RDinoB + NBeuph) - 
                                                 (mDinoB)*DinoB + 0.01)
                                     }
                                     
)}
dCynB <-  function(t,x,params){with( c(x,params), 
                                   aCynB*(NAeuph + NBeuph)*CynB/(RCynB + NAeuph + NBeuph) - 
                                     (mCynB)*CynB + 0.01
)}
```





```{r}
dF <- function(t,x,params = params) {
  
  out <- c(
    NAdeep <- dNAdeep(t,x,params),
    NAeuph <- dNAeuph(t,x,params),
    NBdeep <- dNBdeep(t,x,params),
    NBeuph <- dNBeuph(t,x,params),
    CynB <- dCynB(t,x,params),
    DinoA <- dDinoA(t,x,params),
    DinoB <- dDinoB(t,x,params)
    )

  return(list(out))
}
```



```{r}
burn <- 1000
ts_length <- 3000
tau <- 10
ts <- seq(1,burn+ts_length)*tau

params$mld_critA <- quantile(vapply(ts,fEA,1),0.05)
params$mld_critB <- quantile(vapply(ts,fEB,1),0.06)

```


```{r}
x0 <- c(NAdeep=2,NAeuph=2,NBdeep=.5,NBeuph=.5,CynB=.1,DinoA=.2,DinoB=.05)



block_raw <- ode(y = x0,times = ts,func = dF,parms = params)
```

```{r}
df_model <- as.data.frame(block_raw)
names(df_model) <- c('day',names(x0))

df_model <- df_model %>%
  filter(day > burn*10) %>%
  mutate(mld = fEA(day)) %>%
  mutate(wind = fEB(day)) %>%
  filter(day %% 10 == 0) %>%
  mutate(day = day/10)

```


```{r}
ggplot(df_model,aes(x=day)) + geom_line(aes(y=DinoA,col='Dino A')) + geom_line(aes(y=DinoB,col='Dino B')) + theme_bw() + labs(y='Abundance')
```


```{r}
ggplot(df_model,aes(x=day,y=DinoA+DinoB+.02*CynB)) + geom_line() + labs(y='chlorophyll')
```


```{r}
ggplot(df_model,aes(x=DinoA,y=DinoB)) + geom_point()
```





Make two blocks, one which has species dynamics resolved, the other just has "chlorophyll".


```{r,eval=FALSE}
df_model %>%
  mutate(chl = DinoA + DinoB + .02*CynB) %>%
  select_('day','chl','NAeuph','NBeuph','mld','wind') %>%
  write.csv(file='monod_2box_2bloom.chl.csv',row.names=FALSE,quote=FALSE)

df_model %>%
  write.csv(file='monod_2box_2bloom.species.csv',row.names=FALSE,quote=FALSE)
```
