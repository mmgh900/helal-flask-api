import json
import numpy as np
import pymssql
from flask import Flask, request, jsonify, abort, Request
from flask_cors import CORS
import codecs
from ai_models.days_open import DaysOpenModel
from ai_models.insemination_res import InseminationResModel

app = Flask(__name__)
CORS(app)

# Define a function to create a database connection
def create_connection():
    server = "192.168.101.246\\SQLEXPRESS"
    database = "ModiranFarmer"
    username = "mmgh900"
    password = "0936"
    conn = pymssql.connect(server, username, password, database)

    return conn



days_open_model = DaysOpenModel(create_connection())
insemination_res_model = InseminationResModel(create_connection())

# Read histogram keys and queries from JSON file in Windows-1256 encoding
with codecs.open("histogram_config.json", "r", "UTF-8") as file:
    histogram_config = json.load(file)

def create_histogram(bins: int, query: str):
    # Initialize database connection at startup
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute(query)
    rows = cursor.fetchall()

    # Convert the values in the AllMilk column to float, filtering out any None values
    data = [float(row[0]) for row in rows if row[0] is not None and not np.isnan(float(row[0]))]

    # Create histogram bins using NumPy
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_edges = np.ceil(bin_edges)

    # Create a histogram data structure
    histogram = {
        "data": hist.tolist(),
        "bins": bin_edges.tolist()
    }
    conn.close()
    return histogram


@app.route("/histograms")
def get_histogram_names():
    names = [histogram["name"] for histogram in histogram_config]
    return jsonify({"histogram_names": names})

# Equivalent of get_histogram_params
@app.route("/histograms/<string:histogram_key>/params")
def get_histogram_params(histogram_key):
    params = [histogram["parameters"] for histogram in histogram_config if histogram['name'] == histogram_key]
    return jsonify({"histogram_parameters": params[0]})


@app.route("/histograms/<string:histogram_key>")
def get_histogram(histogram_key):
    query = ""
    parameters = {}

    for histogram in histogram_config:
        if histogram["name"] == histogram_key:
            query = histogram["queries"]
            parameters = histogram["parameters"]
            break

    if query == "":
        return jsonify({"error": "Invalid histogram key"})

    missing_params = [param['title'] for param in parameters if param['title'] not in request.args]
    if missing_params:
        error_message = f"Missing required parameters: {', '.join(missing_params)}"
        return jsonify({"error": error_message})

    # Replace query placeholders with actual parameter values
    for param in parameters:
        param_value = request.args.get(param['title'])
        query = query.replace(f"{{{param['title']}}}", str(param_value))

    bins = int(request.args.get('bins', 10))
    histogram = create_histogram(bins, query)

    return jsonify(histogram)


@app.post("/models/days-open/retrain")
async def retrain_o():
    global days_open_model
    days_open_model = DaysOpenModel(create_connection(), force_retrain=True)
    return days_open_model.accuracy

@app.post("/models/insemination-result/retrain")
async def retrain_i():
    global insemination_res_model
    insemination_res_model = InseminationResModel(create_connection(), force_retrain=True)
    return insemination_res_model.accuracy


@app.get("/models/days-open")
def get_milk_production_histogram():
    serial = request.args.get("serial")
    # Initialize database connection at startup
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute(f"""
    SELECT s.Serial, s.LastBriding, s.TalghihStatus, s.SickStatus, s.CowKind, m.MilkPeriod, m.IsDieEmengercy, s.BodyScore, m.Exist
    FROM [ModiranFarmer].[dbo].[Main] m
    RIGHT OUTER JOIN [ModiranFarmer].[dbo].[CowStatus] s ON m.Serial = s.Serial
    WHERE m.Serial = '{serial}'
    ORDER BY m.MilkPeriod DESC
    """)
    rows = cursor.fetchall()
    if len(rows) < 1:
        abort(404, "شناسه نامعتبر است. این دام در دیتابیس وجود ندارد.")

    status = str(rows[0][8]).strip()
    # if 'حذف شده' in status:
    #     raise HTTPException(status_code=400, detail="این گاو حدف شده است")

    prediction = days_open_model.predict(serial)
    respond = {"data": {
        "Prediction": str(prediction),
        "LastBriding": rows[0][1],
        "TalghihStatus": rows[0][2],
        "SickStatus": rows[0][3],
        "CowKind": rows[0][4],
        "MilkPeriod": rows[0][5],
        "IsDieEmengercy": rows[0][6],
        "BodyScore": rows[0][7],
        "Exist": rows[0][8],

    }}
    conn.close()
    return jsonify(respond)


@app.get("/models/insemination-result")
def get_insemination_result():
    # Initialize database connection at startup
    conn = create_connection()
    cursor = conn.cursor()
    serial = request.args.get("serial")
    date = request.args.get("date")
    natural_breeding = request.args.get("natural_breeding")


    cursor.execute(f"""
    SELECT s.Serial, s.LastBriding, s.TalghihStatus, s.SickStatus, s.CowKind, m.MilkPeriod, m.IsDieEmengercy, s.BodyScore, m.Exist
    FROM [ModiranFarmer].[dbo].[Main] m
    RIGHT OUTER JOIN [ModiranFarmer].[dbo].[CowStatus] s ON m.Serial = s.Serial
    WHERE m.Serial = '{serial}'
    ORDER BY m.MilkPeriod DESC
    """)
    rows = cursor.fetchall()
    if len(rows) < 1:
        abort(404, "شناسه نامعتبر است. این دام در دیتابیس وجود ندارد.")

    status = str(rows[0][8]).strip()
    # if 'حذف شده' in status:
    #     raise HTTPException(status_code=400, detail="این گاو حدف شده است")

    prediction = insemination_res_model.predict(serial, date, natural_breeding == 'true')
    respond = {"data": {
        "Prediction": str(prediction[0]),
        "TimeBred": str(prediction[1].CurrentTimesBred[0] + 1),
        "LastBriding": rows[0][1],
        "TalghihStatus": rows[0][2],
        "SickStatus": rows[0][3],
        "CowKind": rows[0][4],
        "MilkPeriod": rows[0][5],
        "IsDieEmengercy": rows[0][6],
        "BodyScore": rows[0][7],
        "Exist": rows[0][8],

    }}
    conn.close()
    return respond


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
