import numpy as np
import pandas as pd


# Use this function to process the data
# use bld1.csv for the first building.
# use bld2.csv for the second building.
def process_data(filename):
    data = pd.read_csv(filename)
    # Setting the data / time as the index:
    data = data.set_index(data.columns[0])

    # weirdly the data has , instead of . for the decimal separator
    def replace_comma(val):
        new_val = str(val).replace(",", ".")
        return float(new_val)

    # Some of the columns don't change, so we can drop them
    if "bld1.csv" in filename:
        data = data.drop(
            columns=[
                "LIVING ZONE AIR TERMINAL 1:Zone Air Terminal Outdoor Air Volume Flow Rate [m3/s](TimeStep)",
                "LIVING ZONE AIR TERMINAL 2:Zone Air Terminal Outdoor Air Volume Flow Rate [m3/s](TimeStep)",
                "LIVING ZONE:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)",
                "LIVING ZONE:Zone Thermostat Heating Setpoint Temperature [C](TimeStep)"
            ]
        )
    elif "bld2.csv" in filename:
        data = data.drop(
            columns=[
                "ZONE A WINDACFAN:Fan Air Mass Flow Rate [kg/s](TimeStep)",
                "ZONE B WINDACFAN:Fan Air Mass Flow Rate [kg/s](TimeStep)",
                "ZONE C WINDACFAN:Fan Air Mass Flow Rate [kg/s](TimeStep)",
                "ZONE A BASEBOARD:Baseboard Air Mass Flow Rate [kg/s](TimeStep)",
                "ZONE B BASEBOARD:Baseboard Air Mass Flow Rate [kg/s](TimeStep)",
                "ZONE C BASEBOARD:Baseboard Air Mass Flow Rate [kg/s](TimeStep)",
                "LIVING ZONE:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)",
                "LIVING ZONE:Zone Thermostat Heating Setpoint Temperature [C](TimeStep)"
            ]
        )

    cols = data.columns
    # Now we'll move over the dataset and return a processed dataset with floats
    for col in cols:
        data[col] = data[col].apply(replace_comma)

    return cols, data
