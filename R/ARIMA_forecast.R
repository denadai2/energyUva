library(zoo)
library(xts)
library(forecast)
Sys.setenv(TZ = "GMT")
fmt <- '%Y-%m-%d %H:%M:%S'
dat <- read.zoo('../datasets/740-NTH_elektriciteit_withWeather_clean.csv', tz='GMT', format=fmt, header=TRUE, sep=',', stringsAsFactors=FALSE, index.column=9)
dat.xts <- as.xts(dat$gas..m3.)
dat.xts <- dat.xts["/2014-03-31 00:00"]

# Predicted is the forecast variable, predictedHi is the highest 95% confidence forecast
predicted <- vector()
predictedHi <- vector()
# Actual is the variable to help calculating the RMSE
actual <- vector()
dat.ts <- ts(drop(coredata(dat.xts)), frequency=24, start=1)
print(floor(length(dat.xts)/24))
for(i in 0:(floor(length(dat.xts)/24)-21)){ #
	tempTS <- window(dat.ts, start=(i+1), end=(21+i), frequency=24)
	if (i%%10 == 0){
			arima <- auto.arima(tempTS)
		} else {
			arima <- Arima(tempTS, model=arima)
		}
  	print(i)
  	predicted <- append(predicted, forecast(arima, h=24)$mean) 
	predictedHi <- append(predictedHi, forecast(arima, h=24)$upper[,2])

	sliced <- window(dat.ts, start=(21+i), end=(22+i), frequency=24)
  	sliced <- sliced[-length(sliced)]
	actual <- append(actual, sliced)
}
# It forecasts one hour more than necessary
#predicted <- predicted[-length(predicted)]
#predictedHi <- predictedHi[-length(predictedHi)]

export <- xts(predicted, index(dat.xts["2008-01-21 01:00/"]))
colnames(export)[NCOL(export)] <- "predicted"
export$predictedHi <- xts(predictedHi, index(dat.xts["2008-01-21 01:00/"]))
write.zoo(export, "../datasets/ARIMA_forecast.csv", sep=",")





