from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import os

# Heavy optional dependencies (TensorFlow, scikit-learn) may not be installed
# in every environment. Import them defensively so the app can still start and
# provide informative messages rather than crashing at import time.
try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
except Exception:
    StandardScaler = None
    LabelEncoder = None

app = Flask(__name__)

# Load the trained model (try multiple candidate paths)
model = None
model_candidates = [
    'Project/model/model_v4/eeg_movement_model.h5',
    'Project/predict/dqn_wheelchair_model.keras',
]
if tf is None:
    print("TensorFlow not available; model will not be loaded.")
else:
    for p in model_candidates:
        try:
            if os.path.exists(p):
                model = tf.keras.models.load_model(p)
                print(f"Model loaded successfully from: {p}")
                break
        except Exception as e:
            print(f"Failed loading model at {p}: {e}")
    if model is None:
        print("No usable model found in candidate paths.")

# Load and prepare data for preprocessing
data_path = 'Final_clean.csv'
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    scaler = StandardScaler()
    scaler.fit(df[['Value']])
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Result'])
    print("Data loaded and scaler fitted.")
else:
    scaler = None
    label_encoder = None
    print("Data file not found.")

# HTML template for the web app
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brain-Controlled Wheelchair Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        form { margin-bottom: 20px; }
        input { margin: 5px; padding: 8px; }
        button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        #result { margin-top: 20px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Brain-Controlled Wheelchair Predictor</h1>
    <p>Upload a CSV file with EEG data or enter values manually to predict wheelchair movement.</p>

    <form action="/predict" method="post" enctype="multipart/form-data">
        <label for="csv_file">Upload CSV file:</label>
        <input type="file" id="csv_file" name="csv_file" accept=".csv"><br><br>

        <label for="manual_value">Or enter EEG value manually:</label>
        <input type="number" id="manual_value" name="manual_value" step="any"><br><br>

        <button type="submit">Predict</button>
    </form>

    <div id="result">{{ result|safe }}</div>

    <script>
        // Optional: Add JavaScript for real-time updates if needed
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, result="")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or label_encoder is None:
        return render_template_string(HTML_TEMPLATE, result="<p style='color: red;'>Model or data not loaded properly.</p>")

    prediction_result = ""

    # Check if a CSV file was uploaded
    if 'csv_file' in request.files and request.files['csv_file'].filename != '':
        file = request.files['csv_file']
        try:
            df_upload = pd.read_csv(file)
            numeric_cols_check = df_upload.select_dtypes(include=[float, int]).columns.tolist()
            if len(numeric_cols_check) == 0:
                prediction_result = "<p style='color: red;'>CSV must contain numeric columns (e.g., 'Value' or multiple feature columns).</p>"
            else:
                # Try to infer model's expected feature size
                try:
                    model_input_shape = model.input_shape
                    expected_features = int(model_input_shape[-1])
                except Exception:
                    expected_features = 1

                # If the uploaded CSV has >= expected_features numeric columns, use first N columns
                numeric_cols = df_upload.select_dtypes(include=[float, int]).columns.tolist()
                if len(numeric_cols) >= expected_features:
                    feature_cols = numeric_cols[:expected_features]
                    raw_values = df_upload[feature_cols].values
                elif 'Value' in df_upload.columns and expected_features == 1:
                    raw_values = df_upload['Value'].values.reshape(-1, 1)
                else:
                    prediction_result = ("<p style='color: red;'>Uploaded CSV does not contain enough numeric columns "
                                         f"(need {expected_features}). Found columns: {', '.join(numeric_cols)}</p>")
                    return render_template_string(HTML_TEMPLATE, result=prediction_result)

                predictions = []
                for row in raw_values:
                    # row is shape (expected_features,)
                    try:
                        # Apply scaler only if it was fitted on same number of features
                        use_scaler = False
                        if hasattr(scaler, 'n_features_in_'):
                            use_scaler = (scaler.n_features_in_ == expected_features)
                        else:
                            # fallback: if scaler.mean_ exists and matches size
                            use_scaler = (hasattr(scaler, 'mean_') and getattr(scaler, 'mean_').shape[0] == expected_features)

                        arr = np.array(row, dtype=float).reshape(1, -1)
                        if use_scaler:
                            arr = scaler.transform(arr)
                        else:
                            # leave raw values (model may still work)
                            pass

                        # Prepare input shape for model.predict
                        # If model expects a time dimension (e.g., (None, 1, features)), add that axis
                        inp = arr
                        if len(model.input_shape) == 3:
                            # reshape to (1, time_steps, features). If time_steps==1, expand dims
                            time_steps = int(model.input_shape[1]) if model.input_shape[1] is not None else 1
                            if time_steps == 1:
                                inp = arr.reshape(1, 1, expected_features)
                            else:
                                # try to reshape but may fail if lengths mismatch
                                inp = arr.reshape(1, time_steps, expected_features)
                        else:
                            # assume model expects (batch, features)
                            inp = arr.reshape(1, expected_features)

                        pred = model.predict(inp, verbose=0)
                        predicted_class = np.argmax(pred, axis=-1)[0]
                        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
                        predictions.append(predicted_label)
                    except Exception as e:
                        predictions.append(f"<error:{str(e)}>")

                prediction_result = f"<p>Predictions for uploaded data: {', '.join(map(str,predictions))}</p>"
        except Exception as e:
            prediction_result = f"<p style='color: red;'>Error processing CSV: {str(e)}</p>"

    # Check if manual value was entered
    elif 'manual_value' in request.form and request.form['manual_value'] != '':
        mv = request.form['manual_value'].strip()
        try:
            # Accept either a single number or comma/space-separated list of numbers
            if ',' in mv:
                parts = [p.strip() for p in mv.split(',') if p.strip() != '']
            else:
                parts = mv.split()
            nums = [float(p) for p in parts]

            # Determine expected features
            try:
                expected_features = int(model.input_shape[-1])
            except Exception:
                expected_features = 1

            if len(nums) != expected_features:
                prediction_result = (f"<p style='color: red;'>Model expects {expected_features} features. "
                                     f"You provided {len(nums)}. Provide {expected_features} numbers (comma or space separated).</p>")
                return render_template_string(HTML_TEMPLATE, result=prediction_result)

            arr = np.array(nums, dtype=float).reshape(1, -1)

            # Apply scaler only if compatible
            use_scaler = False
            if hasattr(scaler, 'n_features_in_'):
                use_scaler = (scaler.n_features_in_ == expected_features)
            else:
                use_scaler = (hasattr(scaler, 'mean_') and getattr(scaler, 'mean_').shape[0] == expected_features)

            if use_scaler:
                arr = scaler.transform(arr)

            # reshape to match model (handle possible time axis)
            if len(model.input_shape) == 3:
                time_steps = int(model.input_shape[1]) if model.input_shape[1] is not None else 1
                if time_steps == 1:
                    inp = arr.reshape(1, 1, expected_features)
                else:
                    inp = arr.reshape(1, time_steps, expected_features)
            else:
                inp = arr.reshape(1, expected_features)

            pred = model.predict(inp, verbose=0)
            predicted_class = np.argmax(pred, axis=-1)[0]
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
            prediction_result = f"<p>Prediction for value(s) {', '.join(map(str,nums))}: {predicted_label}</p>"
        except ValueError:
            prediction_result = "<p style='color: red;'>Invalid value entered. Provide numbers separated by commas or spaces.</p>"

    else:
        prediction_result = "<p style='color: red;'>Please upload a CSV file or enter a value.</p>"

    return render_template_string(HTML_TEMPLATE, result=prediction_result)

if __name__ == '__main__':
    import os
    # Support both local development and cloud deployment
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.run(debug=debug, use_reloader=False, host=host, port=port)
