library(zoo)
library(xts)
library(forecast)
Sys.setenv(TZ = "GMT")
fmt <- '%Y-%m-%d %H:%M:%S'
dat <- read.zoo('../datasets/740-NTH_elektriciteit_withWeather_clean.csv', tz='GMT', format=fmt, header=TRUE, sep=',', stringsAsFactors=FALSE, index.column=1)
dat.xts <- as.xts(dat$gas..m3.)
dat.xts <- dat.xts["/2014-03-31 00:00"]

dat.ts <- ts(drop(coredata(dat.xts)), frequency=24, start=1)
fit <- stl(dat.ts, "per")
export <- xts(fit$time.series[,"remainder"], index(dat.xts))
colnames(export)[NCOL(export)] <- "remainder"
export$trend <- xts(fit$time.series[,"trend"], index(dat.xts))
write.zoo(export, "../datasets/residuals_day.csv", sep=",")

dat.ts <- ts(drop(coredata(dat.xts)), frequency=(7*24), start=1)
fit <- stl(dat.ts, "per")
export <- xts(fit$time.series[,"remainder"], index(dat.xts))
colnames(export)[NCOL(export)] <- "remainder"
export$trend <- xts(fit$time.series[,"trend"], index(dat.xts))
write.zoo(export, "../datasets/residuals_week.csv", sep=",")

dat.ts <- ts(drop(coredata(dat.xts)), frequency=(365*24), start=1)
fit <- stl(dat.ts, "per")
export <- xts(fit$time.series[,"remainder"], index(dat.xts))
colnames(export)[NCOL(export)] <- "remainder"
export$trend <- xts(fit$time.series[,"trend"], index(dat.xts))
write.zoo(export, "../datasets/residuals_year.csv", sep=",")

# electricity
dat.xts <- as.xts(dat$elektriciteit..kwh.)
dat.ts <- ts(drop(coredata(dat.xts)), frequency=24, start=1)
fit <- stl(dat.ts, "per")
export <- xts(fit$time.series[,"remainder"], index(dat.xts))
colnames(export)[NCOL(export)] <- "remainder"
export$trend <- xts(fit$time.series[,"trend"], index(dat.xts))
write.zoo(export, "../datasets/residuals_electricity_day.csv", sep=",")


