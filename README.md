# 🚴 Food Delivery Time Prediction

This project predicts the estimated food delivery time based on the delivery partner’s age, ratings, and the distance between the restaurant and the delivery location.

It uses an LSTM model trained on historical delivery data and is deployed as a user-friendly Streamlit web application with a clean UI and live map route visualization.

---

## 🧠 Model Overview

- **Model Type:** LSTM (Long Short-Term Memory Neural Network)
- **Input Features:**
  - Delivery Person Age
  - Delivery Person Rating
  - Distance (calculated from latitude & longitude using the Haversine formula)
- **Output:** Estimated delivery time in minutes

---

## 🌐 App Features

- ✅ Real-time delivery time predictions  
- 📍 Haversine distance calculation from coordinates  
- 🗺️ Route visualization with PyDeck map  
- 🎈 Streamlit animations and balloon effect  
- 🖼️ Stylish background and semi-transparent UI overlay

---

## 🚀 Live Demo

> Coming soon after deployment to [Streamlit Cloud](https://streamlit.io/cloud)

---

## 🛠️ Installation

```bash
git clone https://github.com/J-Kimani/Food-delivery_time.git
cd Food-delivery_time
pip install -r requirements.txt
streamlit run food_app.py
```

Make sure the file `food_delivery_model.h5` (trained LSTM model) is placed in the root directory.

---

## 📦 Dependencies

```text
streamlit
numpy
pandas
keras
pydeck
```

To install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```text
📦 Food-delivery_time/
├── food_app.py              # Streamlit web app
├── food_delivery_model.h5   # Trained LSTM model
├── README.md
├── requirements.txt
```

---

## ✍️ Author

**Joshua Kimani** — [GitHub Profile](https://github.com/J-Kimani)

---

## 📌 License

This project is licensed under the MIT License.
