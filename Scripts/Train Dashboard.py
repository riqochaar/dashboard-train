# Import packages
import pandas as pd
import numpy as np
import random
import itertools
import math
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import datetime

# constants
number_of_trips = 2000000
avg_train_speed = 70

# Generate an exponential cumulitve distribution function and hence individual weightings based on lambda and number of points
def exponential_cdf_list(lambd, num_points, max_gap):
    u = np.random.uniform(size=num_points)
    x = np.log(1 - u) / lambd
    x_sorted = np.sort(x)
    x_normalized = (x_sorted - np.min(x_sorted)) / (np.max(x_sorted) - np.min(x_sorted))
    x_list = x_normalized.tolist()
    
    # Adjust gaps between numbers
    y_list = [x_list[i+1] - x_list[i] for i in range(len(x_list)-1)]
    adjusted_y_list = []
    for gap in y_list:
        adjusted_gap = min(gap, max_gap)
        adjusted_y_list.append(adjusted_gap)
    
    # Normalize adjusted gaps
    sum_adjusted_gaps = sum(adjusted_y_list)
    normalized_adjusted_y_list = [gap / sum_adjusted_gaps for gap in adjusted_y_list]
    
    return normalized_adjusted_y_list

# Calculate distance between two coordinates
def calculate_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Radius of the Earth in kilometers
    radius = 6371

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = radius * c

    return distance

# Adjust price for peak travel
def multiply_within_time_range(time, price, peak_multiplier):
    hour = time.hour
    minute = time.minute
    is_within_range = (
        (6 <= hour < 9 and 0 <= minute <= 59) or
        (16 <= hour < 19 and 0 <= minute <= 59)
    )
    if is_within_range:
        return price * peak_multiplier
    else:
        return price

def potential_to_skip(date, day, probability):
    
    if date.weekday() == day:  
        skip_probability = random.random()
        if skip_probability > probability:
            skip = 1
            return skip
        
def potential_to_change(day, month, months_list, day_min, day_max):
    
    if month in months_list:
        change_probability = random.random()
        if change_probability > 0.9:
            day_new = random.randint(day_min, day_max)
        else: 
            day_new = day 
    else:
        day_new = day      
    return day_new

def generate_datetime_list(x):
    
    datetimes = []
    months_with_31_days = [1, 3, 5, 7, 8, 10, 12]
    months_with_30_days = [4, 6, 9, 11]
    while len(datetimes) < x:
        year = random.randint(2021, 2022)
        month = random.choices(range(1, 13), weights=[2, 2, 2, 3, 3, 3, 4, 4, 4, 3, 3, 2])[0]
        day = random.randint(1, 28)  # Assuming all months have 28 days for simplicity
        # Changing some days to 29th, 30th, or 31st depending on months
        day = potential_to_change(day, month, months_with_31_days, 29, 31)
        # Changing some days to 29th or 30th depending on months
        day = potential_to_change(day, month, months_with_30_days, 29, 30)
        date = datetime.date(year, month, day)
        # Weighting for days: e.g., Making monday less popular than Thursday
        skip = potential_to_skip(date, 0, 0.2)
        if skip == 1:  
            continue  
        skip = potential_to_skip(date, 1, 0.8)
        if skip == 1:  
            continue  
        skip = potential_to_skip(date, 2, 0.7)
        if skip == 1: 
            continue
        skip = potential_to_skip(date, 0, 0.3)
        if skip == 1:
            continue  
        # Setting time weekends for weekdays
        if date.weekday() < 5:  # Weekday 
            hour = random.choices([5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23], weights=[2, 5, 8, 6, 2, 2, 5, 8, 6, 2, 2, 1, 1])[0]
        else:  # Weekend 
            hour = random.randint(0, 23)  
        minute = random.randint(0, 59)            
        if not (0 <= hour <= 4):  
            datetime_obj = datetime.datetime(year, month, day, hour, minute, 0)  
            datetimes.append(datetime_obj)
    
    return datetimes

# Check if number is between two other numbers
def between_two_numbers(num, a, b):
    
    if a < num and num < b: 
        
        return True
    
# Generate additive probabilities from a set of items and their probabilities: 
# e.g. convert [0.1, 0.2, 0.3, 0.4] to [0, 0.1, 0.3, 0.6, 1.0]
def generate_additive_probabilities(dictionary):
    
    items = list(dictionary.keys())
    
    probabilities = list(dictionary.values())

    probabilities_additive = [0]

    for p in range(0, len(probabilities)):

        numbers_to_add = probabilities[:p+1]

        total = sum(numbers_to_add)

        probabilities_additive.append(total)
        
    return probabilities_additive
    
# Generate random numbers and hence random items
def generate_random_item(items, probabilities_additive):

    random_number = random.random()

    check_where_number_is_on_numberline = [between_two_numbers(random_number, probabilities_additive[i - 1], probabilities_additive[i]) 
                             for i in range(1, len(probabilities_additive))]
    
    index_of_item = check_where_number_is_on_numberline.index(True)
    
    item = items[index_of_item]
    
    return item

def generate_dataframe_column_of_random_items_from_probabilities(dictionary, number_of_entries):
    
    items = list(dictionary.keys())
    
    probabilities_additive = generate_additive_probabilities(dictionary)
    
    list_generate_random = [generate_random_item(items, probabilities_additive) for _ in range(number_of_entries)]

    series_generate_random = pd.Series(list_generate_random)
    
    return series_generate_random

def prob_delay(duration_prob_delay, dictionary):
    
    random_1 = random.random()
    
    if random_1 <= duration_prob_delay:
        
        random_2 = random.random()
        
        items = list(dictionary.keys())
    
        probabilities_additive = generate_additive_probabilities(dictionary)
    
        delay = generate_random_item(items, probabilities_additive)
        
    else:
        
        delay = 'No Delay'
        
    return delay

def categorize_peak_time(time):

    # Define the peak time ranges
    peak_time_range_1 = datetime.time(6, 0)
    peak_time_range_2 = datetime.time(9, 0)
    peak_time_range_3 = datetime.time(16, 0)
    peak_time_range_4 = datetime.time(19, 0)

    # Check if the time falls within the peak time ranges
    if (time >= peak_time_range_1 and time < peak_time_range_2) or \
       (time >= peak_time_range_3 and time < peak_time_range_4):
        return "Peak"
    else:
        return "Not Peak"
    
def get_stations(station_1, station_2):

    stations_list = sorted(list((station_1, station_2)))
    
    stations_string = stations_list[0] + ' & ' + stations_list[1]
    
    return stations_string

def get_probability_trip_end(trip_start):
        
    trip_routes = df_routes[df_routes['station_pair'].str.contains(trip_start)]
    
    trip_routes = trip_routes[trip_routes['duration'] > 20]
    
    trip_routes = trip_routes.sort_values('distance')
    
    trip_routes['prob'] = trip_routes['distance'].apply(lambda x: 1 / (x / (sum(trip_routes['distance']) + 1)))
    
    trip_routes['prob_2'] = trip_routes['prob'].apply(lambda x: (x / sum(trip_routes['prob'])))
    
    trip_routes = trip_routes.reset_index(drop = True)
    
    trip_routes_station_pairs = list(trip_routes['station_pair'])
    
    trip_routes_stations = [item.replace(trip_start, '') for item in trip_routes_station_pairs]
    
    trip_routes_stations = [item.replace(' & ', '') for item in trip_routes_stations]
    
    trip_routes_stations = [item.strip() for item in trip_routes_stations]
    
    indices_key = [index for index, value in enumerate(trip_routes_stations) if 'London' in value]
    
    trip_routes_modify = trip_routes.loc[indices_key]
    
    trip_routes_modify['prob'] += 50
    
    trip_routes.loc[indices_key] = trip_routes_modify
    
    trip_routes = trip_routes.sort_values('prob', ascending = False)
    
    trip_routes = trip_routes.reset_index(drop = True)
    
    trip_routes['prob'] = trip_routes['prob'].apply(lambda x: (x / sum(trip_routes['prob'])))
    
    trip_routes = trip_routes.drop('prob_2', axis = 1)
            
    return trip_routes

def choose_trip_end(trip_start, station_probs_end_use):
    
    random_2 = random.random()
        
    items = list(station_probs_end_use[trip_start].keys())
    
    probabilities_additive = generate_additive_probabilities(station_probs_end_use[trip_start])
    
    item = generate_random_item(items, probabilities_additive)
        
    item = item.replace(trip_start, '').replace(' & ', '').strip()
    
    return item
    
def class_price(class_, price_before, class_multiplier):
    
    if class_ == 'First Class':
        
        price_new = price_before * class_multiplier
        
    else:
        
        price_new = price_before
        
    return price_new

def delay_reason(delay_class, date, peak, train):
    
    if delay_class == "No Delay":
    
        delay_reason = "No Delay"
        
    else:
        
        check = [0,0,0]
            
        if date.month in [1, 2, 10, 11, 12]:
                        
            check[0] = 1
                    
        if peak == 'Peak' or train == 'V3: The Toby':
        
            check[1] = 1
            
        if train in ['V1: The Thomas', 'V2: The Percy']:
                        
            check[2] = 1

        indices = [index for index, value in enumerate(check) if value == 1]
                                        
        if len(indices) > 0:
        
            value = random.choice(indices)
            
        else:
            
            value = random.choice([0, 1, 2])
            
        if value == 0:
            
            dictionary = {
            "Weather Conditions": 0.5,
            "Obstruction on Track": 0.2,
            "Infrastructure Problems": 0.4
        }



        elif value == 1:
            
            dictionary = {
            "Congestion": 0.6,
            "Technical Issues": 0.3,
            "Staffing Issues": 0.2
            }

        
        elif value == 2:
            
            dictionary = {
            "Mechanical Issues": 0.7,
            "Safety Inspections and Compliance": 0.2,
            "Breakdown": 0.1
            }


        items = list(dictionary.keys())

        probabilities_additive = generate_additive_probabilities(dictionary)

        delay_reason = generate_random_item(items, probabilities_additive)
        
    return delay_reason

class Station:
    def __init__(self, name, city, lat, lon):
        self.name = name
        self.city = city
        self.lat = lat
        self.lon = lon
        
list_stations = [
    Station("London Waterloo", "London", 51.5031, -0.1136),
    Station("London Victoria", "London", 51.4952, -0.1439),
    Station("London Liverpool Street", "London", 51.5174, -0.0813),
    Station("London Bridge", "London", 51.5055, -0.0865),
    Station("London Euston", "London", 51.5284, -0.1338),
    Station("London Paddington", "London", 51.5154, -0.1756),
    Station("London King's Cross", "London", 51.5328, -0.1234),
    Station("London St Pancras International", "London", 51.5322, -0.1269),
    Station("London Stratford", "London", 51.5432, -0.0022),
    Station("Birmingham New Street", "Birmingham", 52.4776, -1.90061),
    Station("Glasgow Central", "Glasgow", 55.8590, -4.25761),
    Station("Leeds", "Leeds", 53.7957, -1.54811),
    Station("Manchester Piccadilly", "Manchester", 53.4774, -2.23131),
    Station("Edinburgh Waverley", "Edinburgh", 55.9520, -3.18901),
    Station("Reading", "Reading", 51.4586, -0.97111),
    Station("Glasgow Queen Street", "Glasgow", 55.8623, -4.25161),
    Station("Bristol Temple Meads", "Bristol", 51.4490, -2.58191),
    Station("Sheffield", "Sheffield", 53.3781, -1.45971),
    Station("Manchester Victoria", "Manchester", 53.4871, -2.24272),
    Station("Birmingham Snow Hill", "Birmingham", 52.4833, -1.90162),
    Station("Brighton", "Brighton", 50.8294, -0.13702),
    Station("Cardiff Central", "Cardiff", 51.4769, -3.17742),
    Station("Nottingham", "Nottingham", 52.9478, -1.14602),
    Station("Liverpool Lime Street", "Liverpool", 53.4076, -2.97792),
    Station("York", "York", 53.9590, -1.08152),
    Station("Newcastle", "Newcastle", 54.9733, -1.61742),
    Station("Glasgow Argyle Street", "Glasgow", 55.8573, -4.25042),
    Station("Edinburgh Haymarket", "Edinburgh", 55.9458, -3.21862),
    Station("Southampton Central", "Southampton", 50.9077, -1.41343),
    Station("Oxford", "Oxford", 51.7539, -1.27043),
    Station("Leicester", "Leicester", 52.6315, -1.12533),
    Station("Bristol Parkway", "Bristol", 51.5136, -2.54203),
    Station("Cambridge", "Cambridge", 52.1949, 0.13723),
    Station("Milton Keynes Central", "Milton Keynes", 52.0344, -0.77483),
    Station("Cardiff Queen Street", "Cardiff", 51.4760, -3.17023),
    Station("Peterborough", "Peterborough", 52.5734, -0.25083),
    Station("Aberdeen", "Aberdeen", 57.1496, -2.09433),
    Station("East Croydon", "Croydon", 51.3756, -0.09264),
    Station("Swindon", "Swindon", 51.5658, -1.78584),
    Station("Southampton Airport", "Southampton", 50.9503, -1.36104),
    Station("Derby", "Derby", 52.9161, -1.46364),
    Station("Exeter St Davids", "Exeter", 50.7290, -3.54334),
    Station("Basingstoke", "Basingstoke", 51.2683, -1.09064),
    Station("Coventry", "Coventry", 52.4006, -1.51264),
    Station("Norwich", "Norwich", 52.6278, 1.30824)
]

dict_stations = {
    "station": [station.name for station in list_stations],
    "city": [station.city for station in list_stations],
    "lat": [station.lat for station in list_stations],
    "lon": [station.lon for station in list_stations]
}

df_stations = pd.DataFrame(dict_stations)

df_stations['weighting'] = exponential_cdf_list(0.5, len(df_stations) + 1, 0.05)

dict_weighting = {}

for index, row in df_stations.iterrows():

    dict_weighting[row['station']] = row['weighting']

df_stations.index.name = 'station_id'

df_stations

# Generate dataframe: routes

df_routes = pd.DataFrame() 

routes = list(itertools.combinations(list(df_stations['station']), 2))

# station_1

df_routes['station_1'] = [x[0] for x in routes]

# station_2

df_routes['station_2'] = [x[1] for x in routes]

# station_tuple

df_routes['station_pair'] = df_routes.apply(lambda x: get_stations(x['station_1'], x['station_2']), axis=1)

# coordinates


df_routes['lat_1'] = df_routes.join(df_stations.set_index('station'), on = 'station_1')['lat']
df_routes['lon_1'] = df_routes.join(df_stations.set_index('station'), on = 'station_1')['lon']
df_routes['lat_2'] = df_routes.join(df_stations.set_index('station'), on = 'station_2')['lat']
df_routes['lon_2'] = df_routes.join(df_stations.set_index('station'), on = 'station_2')['lon']

# distance

df_routes['distance'] = df_routes.apply(lambda x: calculate_distance(x['lat_1'], x['lon_1'], 
                                                              x['lat_2'], x['lon_2']), axis = 1)
# duration

df_routes['duration'] = df_routes['distance'].apply(lambda x: round(60 * (x / avg_train_speed), 0))

# duration_prob_delay

constant = 2
df_routes['duration_prob_delay'] = df_routes['duration'].apply(lambda x: x / (constant * max(df_routes['duration'])))

# drop co-ordinates from dataframe 

df_routes = df_routes.drop(['lat_1', 'lon_1', 'lat_2', 'lon_2'], axis=1)

# station_popularity

df_routes['stations_popularity_1'] = df_routes.join(df_stations.set_index('station'), on = 'station_1')['weighting']
df_routes['stations_popularity_2'] = df_routes.join(df_stations.set_index('station'), on = 'station_2')['weighting']
df_routes['stations_popularity'] = df_routes.apply(lambda x: (x['stations_popularity_1'] + x['stations_popularity_2']) / 2, axis = 1)

df_routes = df_routes.drop(['stations_popularity_1', 'stations_popularity_2'], axis=1)

# ticket price (off-peak)

constant = 5
df_routes['ticket_price'] = df_routes.apply(lambda x: x['duration'] / constant * (1 - (2 * constant * x['stations_popularity'])), axis = 1)

# peak_multiplier

constant = 20
df_routes['peak_multiplier'] = df_routes.apply(lambda x: 1 + (constant * x['stations_popularity']), axis = 1)

# first_class_multiplier

constant = 5
df_routes['first_class_multiplier'] = df_routes.apply(lambda x: 1 + (constant * x['stations_popularity']), axis = 1)

station_probs_end = {}
station_probs_end_use = {}

for station in df_stations['station']:
        
    station_probs_end[station] = get_probability_trip_end(station)
    
    station_probs_end_use[station] = {}
    
    for index, row in station_probs_end[station].iterrows():
        
        station_probs_end_use[station][row['station_pair']] = row['prob']

# Generate dataframe: delays

df_delays = pd.DataFrame() 

train_delays = {
    ("Minor Delay", 1, 5, 0.5),
    ("Moderate Delay", 6, 15, 0.25),
    ("Substantial Delay", 16, 30, 0.12),
    ("Significant Delay", 31, 60, 0.08),
    ("Major Delay", 61, 181, 0.05)
}
    
# delay_class

df_delays['delay_class'] = [x[0] for x in train_delays]
    
# min_lower

df_delays['min_lower'] = [x[1] for x in train_delays]
df_delays['min_lower'] = df_delays['min_lower'].astype(float)
    
# min_upper

df_delays['min_upper'] = [x[2] for x in train_delays]
df_delays['min_upper'] = df_delays['min_upper'].astype(float)

# weighting

df_delays['weighting'] = [x[3] for x in train_delays]

# create df_delays_dict

dict_delays = {}

for index, row in df_delays.iterrows():

    dict_delays[row['delay_class']] = row['weighting']

# Generate dataframe: trips

df_trips = pd.DataFrame()

# trip_start

df_trips['trip_start'] = generate_dataframe_column_of_random_items_from_probabilities(dict_weighting, number_of_trips)
df_trips['city_start'] = df_trips.join(df_stations.set_index('station'), on = 'trip_start')['city']

# trip_end

df_trips['trip_end'] = df_trips['trip_start'].apply(lambda x: choose_trip_end(x, station_probs_end_use))
df_trips['city_end'] = df_trips.join(df_stations.set_index('station'), on = 'trip_end')['city']

# station_tuple

df_trips['station_pair'] = df_trips.apply(lambda x: get_stations(x['trip_start'], x['trip_end']), axis=1)


# Drop rows where trip_start and trip_end are the same or both trip_start and trip_end are in London

df_trips = df_trips[~((df_trips['trip_start'] == df_trips['trip_end']) |
                        (df_trips['trip_start'].str.contains('London') & df_trips['trip_end'].str.contains('London')))]

# date

df_trips['datetime'] = generate_datetime_list(len(df_trips))

df_trips['datetime'] = pd.to_datetime(df_trips['datetime'])

# Split the 'DateTime' column into separate 'Date' and 'Time' columns
df_trips['date_start'] = df_trips['datetime'].dt.date
df_trips['time_start'] = df_trips['datetime'].dt.time

df_trips = df_trips.drop('datetime', axis = 1)

# price

df_trips['price_org'] = df_trips.join(df_routes.set_index('station_pair'), on = 'station_pair')['ticket_price']

df_trips['peak_multiplier'] = df_trips.join(df_routes.set_index('station_pair'), on = 'station_pair')['peak_multiplier']

df_trips['price_incl_peak'] = df_trips.apply(lambda x: multiply_within_time_range(x['time_start'], x['price_org'], x['peak_multiplier']), axis = 1)

df_trips['peak_cost'] = df_trips.apply(lambda x: x['price_incl_peak'] - x['price_org'], axis = 1)

df_trips['peak'] = df_trips['time_start'].apply(lambda x: categorize_peak_time(x))


df_trips = df_trips.drop(['peak_multiplier'], axis=1)

# duration

df_trips[['duration', 'duration_prob_delay']] =  df_trips.join(df_routes.set_index('station_pair'), on = 'station_pair')[['duration', 'duration_prob_delay']]

# delay_class

df_trips['delay_class'] = df_trips['duration_prob_delay'].apply(lambda x: prob_delay(x, dict_delays))

df_trips = df_trips.drop(['duration_prob_delay'], axis=1)

# delay_range

df_trips[['min_lower', 'min_upper']] = df_trips.join(df_delays.set_index('delay_class'), on = 'delay_class')[['min_lower', 'min_upper']]
df_trips['min_lower'].fillna(0, inplace=True)
df_trips['min_upper'].fillna(0, inplace=True)

# delay_actual

df_trips['delay_actual'] = df_trips.apply(lambda x: random.randint(x['min_lower'], x['min_upper']), axis = 1)

# duration

df_trips['duration_actual'] = df_trips['duration'] + df_trips['delay_actual']

# time_end

df_trips['time_end'] = df_trips.apply(lambda x: (datetime.datetime.combine(x['date_start'], x['time_start']) + datetime.timedelta(minutes = x['duration'])).time(), axis = 1)

# delay_percent

df_trips['delay_percent'] = df_trips.apply(lambda x: x['delay_actual'] / x['duration'], axis = 1)

class_ = {
    'First Class' : 0.1,
    'Second Class' : 0.9
}

df_trips['ticket_class'] = generate_dataframe_column_of_random_items_from_probabilities(class_, number_of_trips)

df_trips['first_class_multiplier'] = df_trips.join(df_routes[['station_pair', 'first_class_multiplier']].set_index('station_pair'), on = 'station_pair')['first_class_multiplier']

df_trips['price_incl_peak_and_class'] = df_trips.apply(lambda x: class_price(x['ticket_class'], x['price_incl_peak'], x['first_class_multiplier']), axis = 1)

df_trips['first_class_cost'] = df_trips.apply(lambda x: x['price_incl_peak_and_class'] - x['price_incl_peak'], axis = 1)

# drop columns

df_trips = df_trips.drop(['first_class_multiplier', 'min_lower', 'min_upper'], axis=1)

df_trips = df_trips[df_trips['duration'] > 10]

df_trips = df_trips.reset_index(drop = True)

df_trips.index.name = 'trip_id'

# train

trains = ['V1: The Thomas', 'V2: The Percy', 'V3: The Toby', 'V4: The Emily', 'V5: The Gordon']

df_trips['train'] = random.choices(trains, k=number_of_trips)

# reason_delay

df_trips['delay_reason'] = df_trips.apply(lambda x: delay_reason(x['delay_class'], x['date_start'], x['peak'], x['train']), axis = 1)

df_stations.index.name = 'Station ID'
df_stations = df_stations[['station', 'city', 'lat', 'lon']].rename(columns={
    'station_id':'Station ID', 
    'station':'Station',
    'city':'City',
    'lat':'Latitude',
    'lon':'Longitude'
})

df_routes.index.name = 'Route ID'
df_routes = df_routes[['station_1', 'station_2', 'station_pair', 'distance', 'duration', 'ticket_price', 'peak_multiplier', 'first_class_multiplier']].rename(columns={
    'station_1':'Station A', 
    'station_2':'Station B',
    'station_pair':'Station Pair',
    'distance':'Distance',
    'duration':'Duration',
    'ticket_price':'Ticket Price',
    'peak_multiplier':'Peak Multiplier',
    'first_class_multiplier':'First Class Multiplier'
})

df_delays.index.name = 'Delay ID'
df_delays = df_delays[['delay_class', 'min_lower', 'min_upper']].rename(columns={
    'delay_class':'Delay Class', 
    'min_lower':'Minutes Delayed Lower Limit',
    'min_upper':'Minutes Delayed Upper Limit'
})

df_trips.index.name = 'Trip ID'
df_trips = df_trips[['trip_start', 'city_start', 'trip_end', 'city_end', 'station_pair', 'date_start', 'time_start', 'peak_cost', 'peak', 'delay_class', 'delay_actual','duration_actual', 'time_end', 'ticket_class', 'price_incl_peak_and_class', 'first_class_cost', 'delay_reason', 'train']].rename(columns={
    'trip_start':'Trip Start',
    'city_start':'City Start',
    'trip_end':'Trip End',
    'city_end':'City End',
    'station_pair':'Station Pair',
    'date_start':'Date',
    'time_start':'Time Start',
    'peak_cost':'Peak Cost',
    'peak':'Peak or Not',
    'delay_class':'Delay Class',
    'delay_actual':'Delay',
    'duration_actual':'Duration',
    'time_end':'Time End',
    'ticket_class':'Ticket Class',
    'price_incl_peak_and_class':'Price',
    'first_class_cost':'First Class Cost',
    'train':'Train',
    'delay_reason':'Delay Reason' 

})

start_date = datetime.datetime(2021, 1, 1, 0, 0)
end_date = datetime.datetime(2022, 12, 31, 23, 0)

date_range = pd.date_range(start=start_date, end=end_date, freq='30min')
df_dates = pd.DataFrame(date_range, columns=['DateTime'])

df_trips.to_csv('Data/Trips.csv', index=True)

# Save the DataFrames to an Excel file with multiple sheets
with pd.ExcelWriter('Data/Other.xlsx') as writer:
    df_stations.to_excel(writer, sheet_name='Stations Departing', index=True)
    df_stations.to_excel(writer, sheet_name='Stations Arriving', index=True)
    df_routes.to_excel(writer, sheet_name='Routes', index=True)
    df_delays.to_excel(writer, sheet_name='Delays', index=True)
    df_dates.to_excel(writer, sheet_name='Dates', index=False)

df_dates.to_csv('Data/dates.csv', index=False)




