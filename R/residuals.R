library(zoo)
library(xts)
library(forecast)
Sys.setenv(TZ = "GMT")
fmt <- '%Y-%m-%d %H:%M:%S'
dat <- read.zoo('740-NTH_elektriciteit_withWeather_clean.csv', tz='GMT', format=fmt, header=TRUE, sep=',', stringsAsFactors=FALSE, index.column=9)
dat.xts <- as.xts(dat$gas..m3.)

dat.ts <- ts(drop(coredata(dat.xts)), frequency=24, start=1)
fit <- stl(dat.ts, "per")
export <- xts(fit$time.series[,"remainder"], index(dat.xts))
colnames(export)[NCOL(export)] <- "remainder"
export$trend <- xts(fit$time.series[,"trend"], index(dat.xts))
write.zoo(export, "dailyResiduals.csv", sep=",")

dat.ts <- ts(drop(coredata(dat.xts)), frequency=(7*24), start=1)
fit <- stl(dat.ts, "per")
export <- xts(fit$time.series[,"remainder"], index(dat.xts))
colnames(export)[NCOL(export)] <- "remainder"
export$trend <- xts(fit$time.series[,"trend"], index(dat.xts))
write.zoo(export, "weeklyResiduals.csv", sep=",")

dat.ts <- ts(drop(coredata(dat.xts)), frequency=(365*24), start=1)
fit <- stl(dat.ts, "per")
export <- xts(fit$time.series[,"remainder"], index(dat.xts))
colnames(export)[NCOL(export)] <- "remainder"
export$trend <- xts(fit$time.series[,"trend"], index(dat.xts))
write.zoo(export, "annualResiduals.csv", sep=",")


