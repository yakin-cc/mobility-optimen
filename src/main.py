import pandas as pd
import plotly.express as px
import plotlyshare
import requests
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import unidecode



def clean_time_format(time_str):
    try:
        return pd.to_datetime(time_str, format='%H:%M').time()
    except ValueError:
        try:
            return pd.to_datetime(time_str, format='%H').time()
        except ValueError:
            return np.nan


def clean_comuna_name(name):
    name = unidecode.unidecode(name)  # Remove accents
    name = name.rstrip('.').strip()  # Remove trailing dots and spaces
    return name


def read_excel_data(file_direction, sheet_name, verbose=True):
    if verbose:
        print(f'Parsing data from {file_direction}...')
    data = pd.read_excel(file_direction, sheet_name=sheet_name, dtype=str)

    if verbose:
        print('Processing data...')
    data.columns = data.columns.str.lower().str.replace(' ', '_')

    data['index'] = data['index'].astype(str)
    data['index'] = data['index'].str.replace('.', '').astype(int)
    df['comuna'] = df['comuna'].apply(clean_comuna_name)

    data['pasajeros'] = pd.to_numeric(data['pasajeros'])
    data['item_position'] = pd.to_numeric(data['item_position'])
    data['bp'] = pd.to_numeric(data['bp'])
    data['tiempo_de_viaje_por_persona_(min)'] = pd.to_numeric(data['tiempo_de_viaje_por_persona_(min)'])

    data['latitud'] = pd.to_numeric(data['latitud'].str.replace(',', '.'))
    data['longitud'] = pd.to_numeric(data['longitud'].str.replace(',', '.'))

    data['hora_base'] = data['hora_base'].astype(str)
    data['hora_parada'] = data['hora_parada'].astype(str)
    data['fecha'] = data['fecha'].astype(str)
    data['hora_base'] = pd.to_datetime(data['hora_base'], format='%H:%M').dt.time
    data['hora_parada'] = data['hora_parada'].apply(clean_time_format)
    data['fecha'] = pd.to_datetime(data['fecha'], format='%d-%m-%Y').dt.date
    # Filtering data
    data = data.drop_duplicates(subset=['latitud', 'longitud'])
    data = data[data['tiempo_de_viaje_por_persona_(min)'] <= 75]
    data = data[data['categoria'] == 'TC']

    if verbose:
        print('Done!')
        print(data.info)
    return data


def plot_coordinates(data, destination, cluster, share=False, verbose=True):
    if verbose:
        print('Creating figure...')

    fig = px.scatter_mapbox(
        data,
        lat="latitud",
        lon="longitud",
        hover_name=None,
        hover_data={'index': True, 'latitud': True, 'longitud': True, str(cluster): True},
        zoom=7,
        color=cluster,
        opacity=0.6,
        size_max=10
    )

    fig.update_traces(marker=dict(size=15),
                      hovertemplate="<b>Index: %{customdata[0]}</b><br>Latitude: %{lat}<br>Longitude: %{"
                                    "lon}<br>Cluster: %{customdata[3]}")

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'resources')
    output_file = os.path.join(output_dir, destination + '.html')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig.write_html(file=output_file, auto_open=False)
    if verbose:
        print(f'Plot saved to {output_file}')

    if share:
        fig.show(renderer='plotlyshare')

    if verbose:
        print('Done!')


def get_route_info(start_cords, end_cords, profile='driving', retries=3):
    url = f'http://127.0.0.1:5000/route/v1/{profile}/{start_cords[1]},{start_cords[0]};{end_cords[1]},{end_cords[0]}'
    params = {
        'overview': 'false',
        'geometries': 'geojson',
        'alternatives': 'false',
        'steps': 'false'
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if 'routes' in data:
                route = data['routes'][0]
                return route['distance']  # Distance in meters
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(1)  # Wait a bit before retrying

    print(f"Failed to get route info after {retries} attempts.")
    return np.inf  # Return a large value if the distance cannot be computed


def calculate_distance(i, j, data):
    return i, j, get_route_info((data[i][0], data[i][1]), (data[j][0], data[j][1]))


def parallelized_osrm_distance_matrix(data, verbose=True):
    num_points = len(data)
    matrix = np.zeros((num_points, num_points))
    start_time = time.time()

    if verbose:
        print('\nCalculating distance matrix...')

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculate_distance, i, j, data) for i in range(num_points) for j in
                   range(i + 1, num_points)]

        with tqdm(total=len(futures), desc="Calculating distances") as pbar:
            for future in as_completed(futures):
                i, j, distance = future.result()
                matrix[i, j] = distance
                matrix[j, i] = distance
                pbar.update(1)

    end_time = time.time()
    elapsed_time = end_time - start_time

    if verbose:
        print(f'Distance matrix calculation completed in {elapsed_time:.2f} seconds.')

    return matrix


def osrm_distance_matrix(data, verbose=True):
    num_points = len(data)
    matrix = np.zeros((num_points, num_points))

    start_time = time.time()

    if verbose:
        print('\nCalculating distance matrix...')
        with tqdm(total=num_points * (num_points - 1) // 2, desc="Calculating distances") as pbar:
            for i in range(num_points):
                for j in range(i + 1, num_points):
                    distance = get_route_info((data[i][0], data[i][1]), (data[j][0], data[j][1]))
                    matrix[i, j] = distance
                    matrix[j, i] = distance
                    pbar.update(1)
    else:
        for i in range(num_points):
            for j in range(i + 1, num_points):
                distance = get_route_info((data[i][0], data[i][1]), (data[j][0], data[j][1]))
                matrix[i, j] = distance
                matrix[j, i] = distance

    end_time = time.time()
    elapsed_time = end_time - start_time

    if verbose:
        print(f'Distance matrix calculation completed in {elapsed_time:.2f} seconds.')
    return matrix


def save_dataframe_to_csv(data, filename, folder='resources'):
    # Construct the full path to save the file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(project_root, folder, filename)

    # Save the DataFrame to CSV
    data.to_csv(save_path, index=False)
    print(f'DataFrame saved to {save_path}')


if __name__ == '__main__':
    df = read_excel_data(r'C:\Users\yakin\python-projects\optimen-optimization-mobilization-LATAM\pythonProject\data'
                         r'\data.xlsx', sheet_name='Sheet 1')
    # df = df[['index', 'latitud', 'longitud']]
    plot_coordinates(df, 'clustered_coordinates_comuna', 'comuna')
    plot_coordinates(df, 'clustered_coordinates_vuelo', 'numero_vuelo')

    # df = df.head(10)
    X = df[['latitud', 'longitud']].values
    distance_matrix = parallelized_osrm_distance_matrix(X)
    agglomeration_clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete',
                                                       distance_threshold=15000, compute_full_tree=True)

    print('\nCreating clusters...')
    labels = agglomeration_clustering.fit_predict(distance_matrix)

    df['cluster'] = labels
    print(df)

    plot_coordinates(df, 'clustered_coordinates_filtered', 'cluster', share=True)
    save_dataframe_to_csv(df, 'clustered_data.csv')
