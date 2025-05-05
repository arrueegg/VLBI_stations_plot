import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def df_from_html(file):
    try:
        dfs = pd.read_html(file)
        if dfs:
            df = dfs[0]
            df = df[df.iloc[:, 1].notna()]
            df.reset_index(drop=True, inplace=True)
            return df
        else:
            print("No tables found in the HTML file.")
    except FileNotFoundError:
        print(f"The file '{file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def get_all_sessions():
    # Create a list of file paths for the master files
    years = np.arange(1986, 2024)
    masterfiles = [f"sessions/master{year}.html" for year in years]

    # Append additional file paths for VGOS master files
    years = np.arange(2017, 2020)
    [masterfiles.append(f"sessions/masterVGOS{year}.html") for year in years]

    # Create an empty DataFrame to store all the session data
    all_sessions = pd.DataFrame()

    # Iterate through each file path and read the data into a DataFrame
    for file in masterfiles:
        df = df_from_html(file)
        all_sessions = pd.concat([all_sessions, df], ignore_index=True)

    # Convert the 'Start' column to datetime and extract only the date
    all_sessions['Start'] = pd.to_datetime(all_sessions['Start'])

    return all_sessions

def filter_sessions(sessions_all, start_year, end_year, sessiontypes=None):
    # Filter sessions based on date
    sessions_filtered = sessions_all[sessions_all['Start'].isin(pd.date_range(start=pd.to_datetime(f'{start_year}-01-01'), end=pd.to_datetime(f'{end_year}-12-31')))]
    # Filter sessions based on session type
    sessions_filtered['FirstTwoLetters'] = sessions_filtered['Code'].str[:2]
    # Filter rows where the first two letters are in the 'sessiontypes' list
    sessions_filtered = sessions_filtered[sessions_filtered['FirstTwoLetters'].isin(sessiontypes)]
    sessions_filtered.drop(columns=['FirstTwoLetters'], inplace=True)
    # Filter sessions based on Duration
    #sessions_filtered = sessions_filtered[sessions_filtered['Dur'] == '24:00']
    
    return sessions_filtered

def station_codes_to_names(res_path, station_codes):
    """
    Given a list of station codes, this function returns the corresponding station names. The function reads the station codes and names from an HTML file located at "../madrigal_download/stations.html". It converts all the station codes to uppercase and then looks for the matching station name for each code in the DataFrame obtained from the HTML file. If a matching station name is found, it is added to the list of station names. If a code is not found in the DataFrame, the string "Unknown" is added instead. The final list of station names is returned.
    
    Parameters:
        station_codes (list): A list of station codes.
        
    Returns:
        list: A list of station names corresponding to the given station codes.
    """
    p = res_path.split("/")[:4]
    file = '/scratch/arrueegg/WP1/madrigal_download/stations.html'
    df = pd.read_html(file)[0]
    station_codes = [s.upper() for s in station_codes]
    return [df.loc[df['Code'] == code, 'Name'].values[0] if code in df['Code'].values else 'Unknown' for code in station_codes]

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on the Earth's surface using the Haversine formula.

    Parameters:
        lat1 (float): The latitude of the first point in degrees.
        lon1 (float): The longitude of the first point in degrees.
        lat2 (float): The latitude of the second point in degrees.
        lon2 (float): The longitude of the second point in degrees.

    Returns:
        distance (float): The distance between the two points in kilometers.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6371.0 * c  # Radius of Earth in kilometers

    return distance # in kilometers


# Optional: convert DMS to decimal degrees for lon and lat if necessary
def dms_to_dd_lon(d, m, s):
    sign = np.sign(d)
    dd = abs(d) + abs(m)/60 + abs(s)/3600
    dd = dd - 360 if dd > 180 else dd
    return sign*dd

def dms_to_dd_lat(d, m, s):
    sign = np.sign(d)
    dd = abs(d) + abs(m)/60 + abs(s)/3600
    dd = dd - 180 if dd > 90 else dd
    return sign*dd

# Splitting the coordinates into separate columns
def split_coords(coord):
    d, m, s = coord.split()
    return float(d), float(m), float(s)

def get_coordinates(trf_file):
    column_widths = [5, 8, 18, 21, 43, 55, 67, 75]
    data = []  # to hold the lines of the section
    header = []
    with open(trf_file, 'r') as file:
        record = False
        header_record = True
        for line in file:
            if "Station_description___ Approx_lon_ Approx_lat_ App_h__" in line:
                record = True
            elif "-SITE/ID" in line:
                record = False
                break  # Exit the loop if you don't need to read the rest of the file
            if record:
                if header_record:
                    header.append(line.strip().split())
                    header_record = False
                else:
                    data.append([line[i:j].strip() for i, j in zip([0] + column_widths[:-1], column_widths)])

    # Create DataFrame from the collected data
    df = pd.DataFrame(data, columns=header[0])

    # Ensure all data is treated as string
    df = df.applymap(str)

    # Splitting the station description into name and additional description
    df['Station_name'] = df['Station_description___'].str[:8]
    df['Description'] = df['Station_description___'].str[8:]

    df['Station_name'] = df['Station_name'].str.replace(' ', '', regex=True)


    # Define your 'split_coords' and 'dms_to_dd' functions as they are used here but not defined in your snippet

    df[['Lon_deg', 'Lon_min', 'Lon_sec']] = df['Approx_lon_'].apply(lambda x: pd.Series(split_coords(x)))
    df[['Lat_deg', 'Lat_min', 'Lat_sec']] = df['Approx_lat_'].apply(lambda x: pd.Series(split_coords(x)))

    # Converting heights to float
    df['App_h'] = df['App_h__'].astype(float)

    df['Approx_lon_dd'] = df.apply(lambda x: dms_to_dd_lon(x['Lon_deg'], x['Lon_min'], x['Lon_sec']), axis=1)
    df['Approx_lat_dd'] = df.apply(lambda x: dms_to_dd_lat(x['Lat_deg'], x['Lat_min'], x['Lat_sec']), axis=1)

    df = df.drop(['Station_description___', 'Approx_lon_', 'Approx_lat_', 'App_h__', 'PT', 'T'], axis=1)

    df.rename(columns={'Domes____': 'Domes'}, inplace=True)

    #stations_to_plot = ['KAUAI', 'KASHIMA','AGGO', 'BADARY', 'CRIMEA', 'HOBART']
    stations_to_plot = None
    if stations_to_plot:
        df['plot'] = False
        for station in stations_to_plot:
            df.loc[df['Station_name'] == station, 'plot'] = True
    else:
        df['plot'] = True

    df = df[df['plot']]

    df['technique'] = "VLBI"
    vgos_stations = ["NYALE13N","GGAO12M","HOBART12","WETTZ13N","MACGO12M","RAEGYEB","WETTZ13S","ONSA13NE","RAEGSMAR","WESTFORD","KOKEE12M","KATH12M","ONSA13SW","ISHIOKA"]
    for station in vgos_stations:
        df.loc[df['Station_name'] == station, 'technique'] = "VGOS"

    return df


def plot_stations(df):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.Mercator()})

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE)

    ax.gridlines(draw_labels=True)

    ax.scatter(df['Approx_lon_dd'].values, df['Approx_lat_dd'].values, color='red', transform=ccrs.PlateCarree())

    texts = []
    for i, row in df.iterrows():
        texts.append(plt.text(row['Approx_lon_dd'], row['Approx_lat_dd'], row['Station_name'], ha='center', transform=ccrs.PlateCarree()))

    plt.xlabel('X Axis Label')
    plt.ylabel('Y Axis Label')
    plt.title('Scatter Plot with Annotations')
    plt.show()


def main():
    # path to trf file:
    trf_file = '/scratch2/arrueegg/WP1/VLBI_stations_plot/ivs2023b.trf'
    #load trf file and get station coordinates
    coords = get_coordinates(trf_file)

    sessions_all = get_all_sessions()

    sessions_filtered = filter_sessions(sessions_all, 1986, 2023, ['R1', 'R4', 'VG', 'VT', 'VR'])

    #safe coords to csv
    #coords.to_csv('/scratchcoords.csv', index=False)

    plot_stations(coords)
    print(coords)

if __name__ == "__main__":
    main()