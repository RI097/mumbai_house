from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route( '/predict_home_price', methods=['POST','GET'])
def predict_home_price():
    area = float(request.form['area'])
    ppa = float(request.form['price per area'])
    location = request.form['location']
    bhk = int(request.form['no.of bedrooms'])
    status = int(request.form['new / resale'])
    gym = int(request.form['gymnasium'])
    lift = int(request.form['lift available'])
    carpark = int(request.form['car parking'])
    maintenance = int(request.form['maintenance staff'])
    security = int(request.form['24x7 security'])
    childplayarea = int(request.form['child play area'])
    clubhouse = int(request.form['clubhouse'])
    intercom = int(request.form['intercom'])
    landscape = int(request.form['landscaped gardens'])
    gas = int(request.form['gas connection'])
    games = int(request.form['indoor games'])
    amenities = int(request.form['amenities'])
    joggingtrack = int(request.form['jogging track'])
    pool = int(request.form['swimming pool'])

    response = jsonify({
        'estimated_price': util.get_estimated_price(location, area, ppa, bhk, status,
                                                    gym, lift, carpark,
                                                    maintenance, security, childplayarea,
                                                    intercom, clubhouse, landscape, games,
                                                    gas, joggingtrack, pool, amenities)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response



if __name__ == "__main__" :
    print("Starting Python Flask Server For Home Price Prediction...")
    util.load_saved_artifacts()
    app.run()
