library(AGread)
library(chron)
args <- commandArgs(trailingOnly = TRUE)
print(args[1])
print(args[2])
data <- read_gt3x(args[1])
# special settings
options(digits.secs = 3)
options(digits = 6)

header = "------------ Data File Created By ActiGraph Link IMU 9DOF Sensor ActiLife v6.13.4 Firmware v%s date format M/d/yyyy at 100 Hz  -----------
Serial Number: %s
Start Time %s
Start Date %s
Epoch Period (hh:mm:ss) 00:00:00
Download Time 18:33:27
Download Date %s
Current Memory Address: 0
Current Battery Voltage: %s     Mode = 12
--------------------------------------------------
Accelerometer X,Accelerometer Y,Accelerometer Z,Temperature,Gyroscope X,Gyroscope Y,Gyroscope Z,Magnetometer X,Magnetometer Y,Magnetometer Z
"
firmware = data[["info"]][["Firmware"]]
serial = data[["info"]][["Serial_Number"]]
start_time = strsplit(as.character(data[["info"]][["Start_Date"]]), split = ' ')[[1]][2]
  
start_date = strsplit(as.character(data[["info"]][["Start_Date"]]), split = ' ')[[1]][1]
yy = strsplit(start_date, '-')[[1]][1]

mm = strsplit(start_date, '-')[[1]][2]
dd = strsplit(start_date, '-')[[1]][3]
start_date = paste(mm, dd, yy, sep = '/')
download_date = data[["info"]][["Download_Date"]]
battery_voltage = data[["info"]][["Battery_Voltage"]]


print("done parsing IMU and writing header")
df <- data[["IMU"]]
is.num <- sapply(df, is.numeric)
df[is.num] <- lapply(df[is.num], round, 6)

actual_start_time = strsplit(as.character(df["Timestamp"][[1]][1]), split = ' ')[[1]][2]
list_time <- c(start_time, actual_start_time)
diff_second <- as.numeric(diff(as.difftime(list_time, units = "secs")))[1]
line_to_add <- diff_second * 100 # 100 Hz
print(start_time)
print(actual_start_time)
print(diff_second)
print(line_to_add)

empty_df <- matrix(0 , ncol = 10, nrow = line_to_add)
empty_df <- data.frame(empty_df)

sink(args[2])
cat(sprintf(header,firmware, serial, start_time, start_date, download_date, battery_voltage))
sink()
write.table(empty_df, args[2], append = TRUE, row.names=FALSE, col.names=FALSE, quote = FALSE, sep = ",")
df <- subset(df, select = -Timestamp)
write.table(df,args[2], append=TRUE , row.names=FALSE, col.names=FALSE, quote = FALSE, sep = ",")
print("Done with table")


