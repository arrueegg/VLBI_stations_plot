import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from geopandas import datasets
import cartopy.io.shapereader as shpreader
from adjustText import adjust_text

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

def filter_and_count_sessions(df, start_year, end_year, sessiontypes):

    # Filter sessions based on session type
    df['FirstTwoLetters'] = df['Code'].str[:2]
    # Filter rows where the first two letters are in the 'sessiontypes' list
    df = df[df['FirstTwoLetters'].isin(sessiontypes)]
    df.drop(columns=['FirstTwoLetters'], inplace=True)

    start_date = pd.to_datetime(f'{start_year}-01-01')
    end_date = pd.to_datetime(f'{end_year}-12-31')
    # Convert the 'Start' column to datetime
    df['Start'] = pd.to_datetime(df['Start'])
    
    # Filter the DataFrame by the date range
    mask = (df['Start'] >= start_date) & (df['Start'] <= end_date)
    filtered_df = df[mask]
    
    # Extract station codes and count the sessions
    station_counts = {}
    for stations in filtered_df['Stations']:
        station_list = stations.split()  # Assuming stations are space-separated
        for station in station_list:
            if station in station_counts:
                station_counts[station] += 1
            else:
                station_counts[station] = 1
    
    # Convert the result to a DataFrame
    result_df = pd.DataFrame(list(station_counts.items()), columns=['Station', 'Session_Count'])

    df_names = station_codes_to_names(result_df)
    result_df.Station = [s.upper() for s in result_df.Station]
    result_df = pd.DataFrame(result_df).merge(df_names, left_on='Station', right_on='Code', how='left')
    return result_df

def station_codes_to_names(station_codes):
    """
    Given a list of station codes, this function returns the corresponding station names. The function reads the station codes and names from an HTML file downloaded from: . 
    It converts all the station codes to uppercase and then looks for the matching station name for each code in the DataFrame obtained from the HTML file. 
    If a matching station name is found, it is added to the list of station names. If a code is not found in the DataFrame, the string "Unknown" is added instead. 
    The final list of station names is returned.
    
    Parameters:
        station_codes (list): A list of station codes.
        
    Returns:
        list: A list of station names corresponding to the given station codes.
    """
    file = '/scratch2/arrueegg/WP1/madrigal_download/stations.html'
    df = pd.read_html(file)[0]
    station_codes = [s.upper() for s in station_codes.Station]
    #return [df.loc[df['Code'] == code, 'Name'].values[0] if code in df['Code'].values else 'Unknown' for code in station_codes]
    # join station_codes with df on the 'Code' column
    merged_df = pd.DataFrame(station_codes, columns=['Code']).merge(df, left_on='Code', right_on='Code', how='left')
    merged_df = merged_df[['Code', 'Name']]
    return merged_df

# Function to extract data from the ITRF2020-IVS-TRF.SSC file
def extract_data_from_ivstrf(file_path):
    stations = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip() and not line.startswith("#") and not line.startswith("DOMES") and len(line) > 50:
                parts = line.split()
                if len(parts) >= 12:
                    domes = parts[2]
                    station_name = parts[3]
                    try:
                        lon_deg = float(parts[-7])
                        lon_min = float(parts[-6])
                        lon_sec = float(parts[-5])

                        sign = np.sign(lon_deg)
                        lon = abs(lon_deg) + abs(lon_min)/60 + abs(lon_sec)/3600
                        lon = lon - 360 if lon > 180 else lon
                        lon = sign*lon

                        lat_deg = float(parts[-4])
                        lat_min = float(parts[-3])
                        lat_sec = float(parts[-2])

                        sign = np.sign(lat_deg)
                        lat = abs(lat_deg) + abs(lat_min)/60 + abs(lat_sec)/3600
                        lat = lat - 180 if lat > 90 else lat
                        lat = sign*lat

                        """if 'W' in line:
                            lon = -lon
                        if 'S' in line:
                            lat = -lat"""

                        stations.append([domes, station_name, lat, lon])
                    except (ValueError, IndexError):
                        continue
    return stations

# Function to extract location name from the ITRF2020_VLBI.SSC.txt file
def extract_location_names(file_path):
    locations = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 6 and "VLBI" in parts:
                domes = parts[0]
                index = parts.index("VLBI")
                location_name = " ".join(parts[1:index]).strip()
                if location_name == "La Plata - Arge":
                    location_name = "La Plata"
                if location_name == "Kauai":
                    location_name = "Kokee"
                locations[domes] = location_name.title()
    return locations

def append_tech(df):
    
    df['technique'] = "VLBI"
    vgos_stations = ["NYALE13S","NYALE13N","GGAO12M","HOBART12","WETTZ13N","MACGO12M","RAEGYEB","WETTZ13S","ONSA13NE","RAEGSMAR","WESTFORD","KOKEE12M","KATH12M","ONSA13SW","ISHIOKA"]
    for station in vgos_stations:
        df.loc[df['Station_Name'] == station, 'technique'] = "VGOS"

    return df

def reduce_dataframe(df, technology, min_sessions=50):
    """
    Reduce the DataFrame based on the specified technology and merge rows from the same location.
    
    Args:
    df (pd.DataFrame): The input DataFrame.
    technology (str): The technology to filter by ("VLBI", "VGOS", or "both").
    
    Returns:
    pd.DataFrame: The reduced DataFrame.
    """

    df = df.dropna()
    df = df[~((df['technique'] == 'VLBI') & (df['Session_Count'] < min_sessions))]

    # Group by location and aggregate
    def aggregate_techniques(techniques):
        techniques = list(set(techniques))
        if "VLBI" in techniques and "VGOS" in techniques:
            return "both"
        else:
            return techniques[0]

    
    df = df.groupby('Location_Name').agg({
        'DOMES': 'first',
        'Station_Name': 'first',
        'Latitude': 'first',
        'Longitude': 'first',
        'technique': aggregate_techniques,
        'Session_Count': 'sum'
    }).reset_index()

    # Filter based on technology
    if technology != "all":
        if technology in ["both"]:
            df = df[df['technique'] == technology]
        elif technology in ["VLBI", "VGOS"]:
            df = df[(df['technique'] == technology) | (df['technique'] == "both")]
        else:
            raise ValueError("Technology must be 'VLBI', 'VGOS', 'both' or 'all'")

    return df

def plot_stations(df, technology):
    shpfilename = shpreader.natural_earth(resolution='110m',
                                       category='cultural',
                                       name='admin_0_countries')

    world = gpd.read_file(shpfilename)

    # Change the projection to Mercator (EPSG:3857) or PlateCarree (EPSG:32662)
    epsg_code = 3857
    world_mercator = world.to_crs(epsg=epsg_code)

    # Convert the DataFrame to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
    # Set the CRS for the GeoDataFrame
    gdf.crs = 'EPSG:4326'  # WGS 84
    # Transform the GeoDataFrame to the Mercator projection
    gdf_mercator = gdf.to_crs(epsg=epsg_code)
    
    # Plot the map with the new projection
    fig, ax = plt.subplots(figsize=(10, 10))
    # Set the background color
    ax.set_facecolor('#add8e6')
    # set the map color
    world_mercator.plot(ax=ax, color='#d9e6c7')
    
    # Set the axis limits to focus on specific areas if needed
    if epsg_code == 3857:
        ax.set_xlim([-2e7, 2e7])
        ax.set_ylim([-1.1e7, 1.7e7])
    elif epsg_code == 32662:
        ax.set_xlim([-2e7, 2e7])
        ax.set_ylim([-0.8e7, 0.94e7])

    # Plot the stations
    gdf_VLBI = gdf_mercator[gdf_mercator.technique == "VLBI"]
    gdf_VGOS = gdf_mercator[gdf_mercator.technique == "VGOS"]
    gdf_both = gdf_mercator[gdf_mercator.technique == "both"]
    # Plot markers and store their coordinates
    marker_positions = []
    if len(gdf_VLBI) > 0:
        ax.scatter(gdf_VLBI.geometry.x, gdf_VLBI.geometry.y, color='tab:orange', s=50, label='SX VLBI')
        marker_positions.extend(list(zip(gdf_VLBI.geometry.x, gdf_VLBI.geometry.y)))
    if len(gdf_VGOS) > 0:
        ax.scatter(gdf_VGOS.geometry.x, gdf_VGOS.geometry.y, color='tab:blue', s=50, label='VGOS')
        marker_positions.extend(list(zip(gdf_VGOS.geometry.x, gdf_VGOS.geometry.y)))
    if len(gdf_both) > 0:
        ax.scatter(gdf_both.geometry.x, gdf_both.geometry.y, color='tab:green', s=50, label='SX VLBI and VGOS')
        marker_positions.extend(list(zip(gdf_both.geometry.x, gdf_both.geometry.y)))

    # Add text labels with an initial offset
    texts = []
    for x, y, label, num in zip(gdf_mercator.geometry.x, gdf_mercator.geometry.y, gdf_mercator.Location_Name, gdf_mercator.Session_Count):
        texts.append(ax.text(x, y + 10, label, ha='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)))

    # Convert marker_positions to a list of tuples containing text objects at those positions
    marker_texts = [ax.text(x, y, '   ', fontsize=5, bbox=dict(facecolor='white', alpha=0.0, edgecolor='none', pad=4)) for x, y in marker_positions]
    texts = texts + marker_texts

    # Adjust text to avoid overlap, including marker positions in the adjustment
    adjust_text(texts, add_objects=marker_texts, 
            avoid_points=False, 
            avoid_self=True,
            only_move={'points':'yx', 'texts':'yx'},)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.legend()

    # set title
    if technology == "both":
        title = "Co-located SX VLBI and VGOS stations"
    elif technology == "VLBI":
        title = "SX VLBI stations"
    elif technology == "VGOS":
        title = "VGOS stations"
    elif technology == "all":
        title = "SX VLBI and VGOS stations"
    plt.savefig(f'{title.replace(" ", "_")}_notitle.png', dpi=300, bbox_inches='tight')
    plt.title(title, fontsize=18, fontweight='bold', pad=12)
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    #plt.show()


def main():
    #################################
    #################################
    ####### control variables #######
    technology = "all"     # "VLBI", "VGOS", "both", "all"
    min_sessions = 100       # minimum number of session per station to be plotted
    #################################
    #################################


    #Define file paths
    geodetic_coords_path = "ITRF2020-IVS-TRF.SSC.txt"
    vlbi_file_path = "ITRF2020_VLBI.SSC.txt"

    # Extracting data from ITRF2020-IVS-TRF.SSC
    ivstrf_stations = extract_data_from_ivstrf(geodetic_coords_path)

    # Convert to DataFrame
    ivstrf_columns = ['DOMES', 'Station_Name', 'Latitude', 'Longitude']
    ivstrf_df = pd.DataFrame(ivstrf_stations, columns=ivstrf_columns)

    # Extracting location names from ITRF2020_VLBI.SSC.txt
    location_names = extract_location_names(vlbi_file_path)

    # Add location names to the ivstrf_df based on DOMES number
    ivstrf_df['Location_Name'] = ivstrf_df['DOMES'].map(location_names)

    # load all sessions
    sessions_all = get_all_sessions()

    station_num_sessions = filter_and_count_sessions(sessions_all, 1986, 2023, ['R1', 'R4', 'VG', 'VT', 'VR'])

    ivstrf_df = ivstrf_df.merge(station_num_sessions[['Name', 'Session_Count']], left_on='Station_Name', right_on='Name', how='left')
    # Drop the redundant 'Name' column
    ivstrf_df.drop('Name', axis=1, inplace=True)

    # Save the final DataFrame to a CSV file
    ivstrf_df.to_csv('ivstrf_stations_with_location_names.csv', index=False)

    # modify and filter df for plotting
    ivstrf_df = append_tech(ivstrf_df)

    ivstrf_df = reduce_dataframe(ivstrf_df, technology, min_sessions=min_sessions)

    plot_stations(ivstrf_df, technology)

    

    print("Data extraction and merging complete. The results are saved to 'ivstrf_stations_with_location_names.csv'.")
    
if __name__ == "__main__":
    main()