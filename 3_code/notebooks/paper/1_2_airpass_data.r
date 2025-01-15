
# Load the datasets package (included in base R)
library(datasets)
library(ggplot2)
library(zoo) 

data(AirPassengers)

# Convert wide to long format
AirPassengers_long <- data.frame(
  Month = as.Date(as.yearmon(time(AirPassengers))),
  Passengers = as.numeric(AirPassengers)
)


# Plot the data
plot <- ggplot(AirPassengers_long, aes(x = Month, y = Passengers)) +
  geom_line() +
  labs(title = "Air Passengers over time",
       x = "Month",
       y = "Number of passengers") +
  theme_minimal()

# Print the plot to ensure it is displayed
print(plot)

# save as csv
# print current working directory
print(getwd())
data_folder <- '../../2_data/processed/airline_passenger'
write.csv(AirPassengers_long, file.path(data_folder, 'airline_passenger.csv'), row.names = FALSE)

