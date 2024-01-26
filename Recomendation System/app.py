from flask import Flask, render_template, request
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from joblib import dump, load
import json
import pandas as pd
df = pd.read_csv(r"C:\Users\ACER\Videos\recomender_system\Recomendation System\FixedDataset.csv")
df = df.drop(df.columns[0], axis=1)
df = df.drop_duplicates() #drop duplicates data


app = Flask(__name__)

# Feature engineering and training (contoh, pastikan menyesuaikan dengan model Anda)
features = ['customer_id', 'page_views', 'time_spent', 'price', 'ratings']
target = 'product_id'
train_data = df[features]

# Inisialisasi model KNN
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(train_data)

# Endpoint for recommendation
@app.route('/')
def index(): #the function name is the same as html file name
    selected_columns = ['product_id', 'category', 'price', 'ratings']
    selected_df = df[selected_columns].head(10)
    return render_template('index.html', tables=[selected_df.to_html(classes='data')], titles=selected_df.columns.values, recommended_products=False) #to be shown on website

def get_current_date():
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    day_string = now.strftime("%A")
    return date_string, day_string

@app.route('/recommend', methods=['POST'])
def recommend():
    # Inisialisasi customer_data
    customer_data = None

    if request.method == 'POST':
        # Ambil input ID pelanggan dari formulir HTML
        try:
            customer_id = int(request.form['customer_id'])
        except ValueError:
            return render_template('index.html', error_message="Customer ID must be a valid number.")

        # Dapatkan data pelanggan berdasarkan ID
        customer_data = df[df['customer_id'] == customer_id][features]

        if customer_data.empty:
            return render_template('index.html', error_message="Customer ID not found in the data.")

    if customer_data is not None:
        # Lakukan prediksi menggunakan model
        distances, indices = model.kneighbors(customer_data, n_neighbors=5)
        
        # Ambil indeks produk yang direkomendasikan
        recommended_product_indices = indices.flatten()

        # Ambil informasi produk yang direkomendasikan
        recommended_products = df.iloc[recommended_product_indices].head(5) #top 5 products

        # Konversi ke format JSON
        recommended_products_json = recommended_products.to_json(orient='records')

        return render_template('result.html', recommended_products=json.loads(recommended_products_json)) #show result of recommendation on result.html

    # Jika tidak ada input atau data pelanggan ditemukan
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
