import sys
import pandas as pd
from typing import Tuple
from datetime import datetime, timedelta

# Mapping different physical activity types to generalized activity classes.
mapping = {
    "Sitting_Still": "Sitting",
    "Sitting_With_Movement": "Sitting",
    "Sit_Recline_Talk_Lab": "Sitting",
    "Sit_Recline_Web_Browse_Lab": "Sitting",
    "Sit_Writing_Lab": "Sitting",
    "Sit_Typing_Lab": "Sitting",
    "Standing_Still": "Standing",
    "Standing_With_Movement": "Standing",
    "Stand_Conversation_Lab": "Standing",
    "Lying_Still": "Lying_Down",
    "Lying_With_Movement": "Lying_Down",
    "Lying_On_Back_Lab": "Lying_Down",
    "Lying_On_Right_Side_Lab": "Lying_Down",
    "Lying_On_Stomach_Lab": "Lying_Down",
    "Lying_On_Left_Side_Lab": "Lying_Down",
    "Walking": "Walking",
    "Treadmill_2mph_Lab": "Walking",
    "Treadmill_3mph_Coversation_Lab": "Walking",
    "Treadmill_3mph_Free_Walk_Lab": "Walking",
    "Treadmill_3mph_Drink_Lab": "Walking",
    "Treadmill_3mph_Briefcase_Lab": "Walking",
    "Treadmill_3mph_Phone_Lab": "Walking",
    "Treadmill_3mph_Hands_Pockets_Lab": "Walking",
    "Walking_Fast": "Walking",
    "Walking_Slow": "Walking",
    "Walking_Up_Stairs": "Walking_Up_Stairs",
    "Walking_Down_Stairs": "Walking_Down_Stairs",
    "Cycling_Active_Pedaling_Regular_Bike": "Biking",
    "Stationary_Biking_300_Lab": "Biking",
    "Ab_Crunches_Lab": "Gym_Exercises",
    "Arm_Curls_Lab": "Gym_Exercises",
    "Push_Up_Lab": "Gym_Exercises",
    "Push_Up_Modified_Lab": "Gym_Exercises",
    "Machine_Leg_Press_Lab": "Gym_Exercises",
    "Machine_Chest_Press_Lab": "Gym_Exercises",
    "Treadmill_5.5mph_Lab": "Gym_Exercises",
}


# Function to read and parse data from the actigraph file.
def read_data(filename: str, agd: bool = False) -> Tuple[datetime, pd.DataFrame]:
    """
    Reads the actigraph data file and returns the starting timestamp and the corresponding DataFrame.

    Parameters:
    - filename: Path to the actigraph file.
    - agd: If True, assume a 1-second interval for sampling.

    Returns:
    - A tuple containing the starting timestamp and the actigraph data as a DataFrame.
    """
    sampling_rate = 1  # Default sampling rate
    start_date = None
    start_time = None

    # Open the file and read metadata
    with open(filename) as f:
        line = f.readline()  # Read first line
        parsed = line.split()  # Split the line by spaces

        # Loop through the parsed line to find the sampling rate (marked by 'Hz')
        for i in range(len(parsed)):
            if parsed[i] == "Hz":
                sampling_rate = int(parsed[i - 1])  # Get the value before 'Hz' as the sampling rate
                break

        f.readline()  # Skip second line
        start_time = f.readline().split()[-1]  # Read start time from the third line
        start_date = f.readline().split()[-1]  # Read start date from the fourth line

    # Combine start date and time, and parse it into a datetime object
    start = datetime.strptime(start_date + " " + start_time, "%m/%d/%Y %H:%M:%S")

    # Calculate the time step between each sample
    step = timedelta(seconds=1 / sampling_rate)
    if agd:
        step = timedelta(seconds=1)  # Use 1 second for AGD format

    # Read the data from the file into a DataFrame (skipping the first 10 rows of metadata)
    df = pd.read_csv(filename, skiprows=10, header=0)

    # Add a 'Timestamp' column to the DataFrame with calculated timestamps for each data point
    df["Timestamp"] = [start + i * step for i in range(len(df))]

    return start, df


# Function to add activity labels to the actigraph data based on time intervals.
def add_label_to_actigraph(actigraph, label) -> pd.DataFrame:
    """
    Adds activity labels to the actigraph data based on the time intervals in the label data.

    Parameters:
    - actigraph: The DataFrame containing actigraph data.
    - label: The DataFrame containing labeled activity data with start and stop times.

    Returns:
    - The actigraph DataFrame with added 'Activity' column containing the activity classes.
    """
    actigraph["Activity"] = None  # Add a new 'Activity' column initialized with None

    # Add two additional labels to the data, one for before the data collection and one for after
    first_label_timestamp = label["START_TIME"].iloc[0]
    last_label_timestamp = label["STOP_TIME"].iloc[-1]

    # Assign before data collection label
    actigraph.loc[actigraph["Timestamp"] < first_label_timestamp, "Activity"] = ("Before_Data_Collection")

    # Assign after data collection label
    actigraph.loc[actigraph["Timestamp"] > last_label_timestamp, "Activity"] = ("After_Data_Collection")

    # Iterate over each row in the label DataFrame
    for _, row in label.iterrows():
        start = row["START_TIME"]
        stop = row["STOP_TIME"]

        # Assign the activity label using loc accessor
        actigraph.loc[
            (actigraph["Timestamp"] >= start) & (actigraph["Timestamp"] <= stop),
            "Activity",
        ] = row["ACTIVITY_CLASS"]

    return actigraph


# Main function to merge actigraph data with labels and output the result as a CSV file.
def data_to_csv(input_actigraph: str, input_label: str, output_actigraph: str) -> None:
    """
    Combines actigraph data with activity labels and saves the result as a CSV file.

    Parameters:
    - input_actigraph: Path to the input actigraph file.
    - input_label: Path to the input label file containing activity intervals.
    - output_actigraph: Path where the combined data should be saved as a CSV file.
    """
    # Read actigraph data
    _, actigraph = read_data(input_actigraph)

    # Read label data and map the activity types to the activity classes
    label = pd.read_csv(input_label, parse_dates=["START_TIME", "STOP_TIME"])
    label["ACTIVITY_CLASS"] = [mapping.get(x, None) for x in label["PA_TYPE"]]

    # Add activity labels to the actigraph data
    actigraph = add_label_to_actigraph(actigraph, label)

    # Save the merged data to a CSV file
    actigraph.to_csv(output_actigraph, index=False)


# The main program execution when the script is run directly.
if __name__ == "__main__":
    # Check if all required command line arguments are provided
    if len(sys.argv) != 4:
        print(
            "Usage: python script.py <input_actigraph> <input_label> <output_actigraph>"
        )
        sys.exit(1)

    # Read input and output file paths from command line arguments
    input_actigraph = sys.argv[1]
    input_label = sys.argv[2]
    output_actigraph = sys.argv[3]

    data_to_csv(input_actigraph, input_label, output_actigraph)
