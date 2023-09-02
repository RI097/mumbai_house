import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, area, ppa, bhk, status,
                        gym, lift, carpark,
                        maintenance, security, childplayarea,
                        intercom, clubhouse, landscape, games,
                        gas, joggingtrack, pool, amenities):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = area
    x[1] = ppa
    x[2] = bhk
    x[3] = status
    x[4] = gym
    x[5] = lift
    x[6] = carpark
    x[7] = maintenance
    x[8] = security
    x[9] = childplayarea
    x[10] = intercom
    x[11] = clubhouse
    x[12] = landscape
    x[13] = games
    x[14] = gas
    x[15] = joggingtrack
    x[16] = pool
    x[17] = amenities

    if loc_index >= 0:
        x[loc_index] = 1


    return round(__model.predict([x])[0],2)

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    global __locations

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[18:]  # first few columns are categorical

    global __model
    if __model is None:
        with open('./artifacts/hpp.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('Khar',720,2000,2,1,1,1,0,1,1,0,0,0,0,0,0,1,0,5),'lakhs')


#location, area, ppa, bhk, status, gym, lift, carpark, maintenance, security, childplayarea,
       # intercom, clubhouse, landscape, games, joggingtrack, swimmingpool, amenities,