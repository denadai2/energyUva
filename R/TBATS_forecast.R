library(zoo)
library(xts)
library(forecast)
Sys.setenv(TZ = "GMT")
fmt <- '%Y-%m-%d %H:%M:%S'
dat <- read.zoo('740-NTH_elektriciteit_withWeather_clean.csv', tz='GMT', format=fmt, header=TRUE, sep=',', stringsAsFactors=FALSE, index.column=9)
dat.xts <- as.xts(dat$gas..m3.)

# Tsbat
predicted <- vector()
predictedHi <- vector()
dat.ts <- ts(drop(coredata(dat.xts)), frequency=24, start=1)
print(floor(length(dat.xts)/48))
for(i in 0:(floor(length(dat.xts)/48)-21)){
	tempTS <- window(dat.ts, start=(2*i+1), end=(21+i*2), frequency=24)
	y <- msts(tempTS, seasonal.periods=c(24, 7*24))
	fit <- tbats(y)
  	print(i)
  	predicted <- append(predicted, forecast(fit, h=48)$mean) 
	predictedHi <- append(predictedHi, forecast(fit, h=48)$upper[,2])
}
# It forecasts one hour more than necessary
predicted <- predicted[-length(predicted)]
predictedHi <- predictedHi[-length(predictedHi)]

export <- xts(predicted, index(dat.xts["2008-01-21 01:00/"]))
colnames(export)[NCOL(export)] <- "predicted"
export$predictedHi <- xts(predictedHi, index(dat.xts["2008-01-21 01:00/"]))
write.zoo(export, "TBATS_martedi.csv", sep=",")

