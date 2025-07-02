## 🌱 EcoPredict AI - Greenhouse Gas Emission Forecasting


![ChatGPT Image Jul 3, 2025, 01_18_35 AM](https://github.com/user-attachments/assets/3b09e24a-0cd6-46af-984d-274d629829f9)

**EcoPredict AI** is an advanced web-based application for predicting greenhouse gas (GHG) emissions in supply chains using artificial intelligence and machine learning technologies. Built with **Streamlit** and powered by a **Linear Regression model** trained on **EPA's official supply chain emission data from 2010–2016**, the application provides intelligent predictions for emission factors across various US industries and commodities.

The system features a modern, interactive GUI with **glassmorphism design**, **animated gradients**, and professional styling. It allows users to input parameters such as:

* Greenhouse gas type (carbon dioxide, methane, nitrous oxide, or others)
* Measurement units (kg, tons, CO₂e, etc.)
* Emission factors
* Comprehensive data quality metrics:

  * Reliability
  * Temporal correlation
  * Geographic correlation
  * Technological correlation
  * Data collection quality

User inputs are processed through a **sophisticated preprocessing pipeline**, standardized using scaling, and sent to the ML model for **real-time predictions**. Visual analytics include **interactive Plotly charts** such as gauges, bar charts, and radar plots. The platform also offers **AI-powered sustainability recommendations** based on input trends.

It serves as both an educational tool and practical solution for environmental impact analysis—ideal for researchers, analysts, and organizations aiming to optimize their carbon footprint.

---

## 🌍 Why EcoPredict AI?

🌿 Climate change is one of the most pressing global challenges. Accurate emission forecasting can support better decisions for a greener future. EcoPredict AI helps:

* 📊 Predict GHG emissions by sector or activity
* 🧠 Visualize trends & outcomes
* ⚡ Deliver real-time results via a beautiful interface
* 📈 Encourage data-driven climate action

---

## 🔧 Features

* ✅ AI-powered emission predictions using EPA data
* 💠 Glassmorphism UI with responsive layout
* 📈 Real-time charts: Gauge, Bar, Radar (Plotly-powered)
* 🔎 Data quality scoring system (5 metrics)
* 🧠 AI suggestions for emission reduction
* 📁 Upload and compare industry datasets
* 🛡️ Error handling and validation checks
* 🌐 Deployable via Streamlit Cloud or localhost

---

## 📸 Screenshots
![Screenshot 2025-07-03 011100](https://github.com/user-attachments/assets/1135d00e-f797-446f-bad9-92647c0eeb00)
![Screenshot 2025-07-03 011111](https://github.com/user-attachments/assets/02df404f-bff8-408b-81f8-6e889f5c4bb6)
![Screenshot 2025-07-03 011119](https://github.com/user-attachments/assets/be9bb5f2-c39b-402e-b236-1b15615f7156)
![Screenshot 2025-07-03 011218](https://github.com/user-attachments/assets/356b5711-6f57-4029-98d7-492f0d90a2f2)
![Screenshot 2025-07-03 011235](https://github.com/user-attachments/assets/0617fc78-d269-4ed2-9f84-4288216b5525)
e071fb800)
![Screenshot 2025-07-03 011243](https://github.com/user-attachments/assets/8119d659-815b-45d8-94f6-416b84ebe78c)

## 📈 Visual Insights

EcoPredict AI provides graphical output that enhances interpretation and decision-making. Users receive:

* 📉 **Line Graphs**: Track emission trends across inputs
* 📊 **Bar Charts**: Compare emissions between industries
* 🎯 **Gauge Charts**: Assess overall impact severity
* 🌐 **Radar Charts**: Visualize data quality profiles

All visualizations are interactive and rendered using **Plotly**.

---

## 🧠 Model & Training Details

The AI model was built and trained using the following pipeline:

### 📚 Libraries Used

* **Core Data Processing**:

  * `pandas`, `numpy`
* **Visualization**:

  * `matplotlib.pyplot`, `seaborn`, `plotly`
* **Machine Learning**:

  * `sklearn.model_selection`:

    * `train_test_split`, `GridSearchCV`
  * `sklearn.preprocessing`:

    * `StandardScaler`
  * `sklearn.linear_model`:

    * `LinearRegression`
  * `sklearn.ensemble`:

    * `RandomForestRegressor`
  * `sklearn.metrics`:

    * `mean_squared_error`, `r2_score`
* **Model Persistence**:

  * `joblib`

### 🎯 Process Summary

* Preprocessing: null value handling, scaling
* Feature engineering
* Model training using **Linear Regression** (final model)
* Evaluation using **MSE** and **R²**
* Hyperparameter tuning with **GridSearchCV**
* Models saved as `LR_model.pkl` and `scaler.pkl`

---

## 💻 Tech Stack

| Layer         | Tools Used                                     |
| ------------- | ---------------------------------------------- |
| Backend       | Python, Pandas, Scikit-learn, LinearRegression |
| Frontend/UI   | Streamlit, Plotly                              |
| Deployment    | Streamlit Cloud / Local Host                   |
| Visualization | Plotly, Seaborn, Matplotlib                    |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/EcoPredict-AI.git
cd EcoPredict-AI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

---

## 🌐 Live Demo

Check out the app live here 👉 [EcoPredict AI - Streamlit App](https://echopredict-ai.streamlit.app/)

---

## 🗂 Project Structure

```
EcoPredict-AI/
├── app.py                 # Streamlit App Logic
├── model/
│   └── LR_model.pkl       # Trained Linear Regression Model
├── scaler/
│   └── scaler.pkl         # StandardScaler instance
├── data/
│   └── emissions.csv      # Industry Emissions Data
├── assets/
│   └── ecopredict_banner.png  # Project banner image
├── utils/
│   └── preprocess.py      # Data Cleaning & Helper Functions
├── requirements.txt
└── README.md
```

---

## 📊 Example Use Case

> **Input:** Industry: "Steel Manufacturing", GHG: "Methane", Unit: "Ton"
> **Prediction:** \~4.82 metric tons CO₂e emitted per ton produced.

Gauge, radar, and bar charts will help interpret the output visually.

---

## 🤝 Contribution Guidelines

We welcome all contributions, ideas, and suggestions!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Licensed under the **MIT License**.

Copyright (c) 2025 Aayush Kumar

Permission is hereby granted to use, copy, modify, and distribute this software for any purpose with or without fee, provided that the above copyright notice appears in all copies.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.


---

## 🙏 Acknowledgements

* EPA Supply Chain GHG Emissions Data
* IBM Greenhouse Gas Project
* Streamlit Community
* Scikit-learn Contributors
* OpenAI Copilot

---

## 👨‍💻 Author

**Aayush Kumar**
📫 Email: \[[aayush05.af@gmail.com](mailto:aayush05.af@gmail.com)]
🔗 LinkedIn: [linkedin.com/in/aayush-kumar-146252314](https://www.linkedin.com/in/aayush-kumar-146252314/)

---

> "The best way to predict the future is to design it sustainably." 🌏
