from flask import Flask, render_template, request
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
import pandas as pd
df1 = pd.read_csv("customer_interactions.csv")
df2 = pd.read_csv("product_details.csv", delimiter=";")
df3= pd.read_csv("purchase_history.csv", delimiter=";")
df_merge= pd.merge(df1, df3.iloc[:, :3], on='customer_id', how='inner') 
df_merge = pd.merge(df_merge, df2.iloc[:, :4], on='product_id', how='inner')
print(df_merge)


app = Flask(__name__)

# Feature engineering and training (contoh, pastikan menyesuaikan dengan model Anda)
features = ['customer_id', 'page_views', 'time_spent', 'price', 'ratings']
target = 'product_id'

X_train, X_test, y_train, y_test = train_test_split(df_merge[features], df_merge[target], test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Simpan model
dump(model, 'recommender_model.joblib')

# Endpoint for recommendation
@app.route('/')
def index(): #the function name is the same as html file name
    selected_columns = ['product_id', 'category', 'price', 'ratings']
    selected_df = df_merge[selected_columns]
    return render_template('index.html', tables=[selected_df.to_html(classes='data')], titles=selected_df.columns.values) #to be shown on website

def get_current_date():
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    day_string = now.strftime("%A")
    return date_string, day_string

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        # Ambil input ID pelanggan dari formulir HTML
        try:
            customer_id = int(request.form['customer_id'])
        except ValueError:
            return render_template('index.html', error_message="Customer ID must be a valid number.")

        # Dapatkan data pelanggan berdasarkan ID
        customer_data = df_merge[df_merge['customer_id'] == customer_id][features]

        if customer_data.empty:
            return render_template('index.html', error_message="Customer ID not found in the data.")

        # Prediksi dengan model
        predictions = model.predict(customer_data)
        
        # Ambil dua produk teratas
        top_products_indices = predictions.argsort()[-3:][::-1]
        top_products = df_merge.iloc[top_products_indices]

        selected_columns = ['product_id', 'category', 'price', 'ratings']
        selected_df = df_merge[selected_columns]

        return render_template('index.html', top_products=top_products.to_dict(orient='records'),tables=[selected_df.to_html(classes='data')], titles=selected_df.columns.values)

if __name__ == '__main__':
    app.run(debug=True)