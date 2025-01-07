from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the machine learning model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        store = int(request.form['Store'])
        day_of_week = int(request.form['DayOfWeek'])
        customers = int(request.form['Customers'])
        open_store = int(request.form['Open'])
        promo = int(request.form['Promo'])
        state_holiday = int(request.form['StateHoliday'])
        school_holiday = int(request.form['SchoolHoliday'])
        store_type = int(request.form['StoreType'])
        assortment = int(request.form['Assortment'])
        competition_distance = float(request.form['CompetitionDistance'])
        competition_open_since_month = int(request.form['CompetitionOpenSinceMonth'])
        competition_open_since_year = int(request.form['CompetitionOpenSinceYear'])
        promo2 = int(request.form['Promo2'])
        promo2_since_week = int(request.form['Promo2SinceWeek'])
        promo2_since_year = int(request.form['Promo2SinceYear'])
        promo_interval = int(request.form['PromoInterval'])
        day = int(request.form['Day'])
        week_of_year = int(request.form['WeekOfYear'])
        month = int(request.form['Month'])
        year = int(request.form['Year'])
        is_weekend = int(request.form['IsWeekend'])
        is_beginning_of_month = int(request.form['IsBeginningOfMonth'])
        is_mid_month = int(request.form['IsMidMonth'])
        is_end_of_month = int(request.form['IsEndOfMonth'])

        # Prepare input for the model
        input_features = np.array([[store, day_of_week, customers, open_store, promo, state_holiday,
                                    school_holiday, store_type, assortment, competition_distance,
                                    competition_open_since_month, competition_open_since_year, promo2,
                                    promo2_since_week, promo2_since_year, promo_interval, day,
                                    week_of_year, month, year, is_weekend, is_beginning_of_month,
                                    is_mid_month, is_end_of_month]])

        # Make prediction
        prediction = model.predict(input_features)[0]

        # Render the result.html template
        return render_template('result.html', store=store, day_of_week=day_of_week, customers=customers,
                               open_store=open_store, promo=promo, state_holiday=state_holiday,
                               school_holiday=school_holiday, store_type=store_type, assortment=assortment,
                               competition_distance=competition_distance, competition_open_since_month=competition_open_since_month,
                               competition_open_since_year=competition_open_since_year, promo2=promo2,
                               promo2_since_week=promo2_since_week, promo2_since_year=promo2_since_year,
                               promo_interval=promo_interval, day=day, week_of_year=week_of_year, month=month,
                               year=year, is_weekend=is_weekend, is_beginning_of_month=is_beginning_of_month,
                               is_mid_month=is_mid_month, is_end_of_month=is_end_of_month, prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=os.environ.get('PORT', 3000))
