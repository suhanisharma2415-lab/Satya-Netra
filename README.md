# 🔍 Satya-Netra: AI-Based Fake News Detection System

Satya-Netra is an advanced, end-to-end software platform designed to identify misleading, false, and fabricated digital news articles automatically. By leveraging Natural Language Processing (NLP) techniques and optimized Machine Learning architectures, the system provides real-time authenticity classifications along with a precise prediction confidence score to enhance digital trust and communication reliability.

🌐 **Live Demo:** [Click Here to View the Live Project Dashboard](https://satya-netra.streamlit.app)

 📌 Objectives & Problem Statement
* **Detect Misinformation:** Automatically parse textual content to identify fake or heavily manipulated news updates.
* **Improve Digital Trust:** Empower everyday users with an intuitive, seamless interface to cross-verify online articles before sharing them.
* **Real-Time Assessment:** Provide high-speed textual evaluation and output instantaneous classification results with reliability percentages.

 🛠️ System Architecture & Workflow
The platform is engineered completely as a modern, software-driven text processing pipeline:

1.  **Data Acquisition:** Aggregates unstructured news content utilizing unified benchmark collections (Kaggle Datasets & public archives).
2.  **Data Preprocessing:** Implements structured linguistic cleaning routines, including case normalization, tokenization, and contextual stopword filtration.
3.  **Feature Extraction:** Translates normalized text blocks into dense numerical vectors using TF-IDF matrices and text embeddings.
4.  **Inference Engine:** Evaluates vector inputs through statistical models (Logistic Regression) and deep learning contextual transformers (BERT).
5.  **Output Layer:** Returns the categorical prediction classification alongside a calculated confidence probability score.


🚀 Key Features
* **Automated Pipeline:** Smooth, sequential data conversion from raw string inputs to processed classification vectors.
* **Contextual NLP Analysis:** Goes beyond simple keyword matching to study underlying contextual syntax and linguistic anomalies.
* **Interactive Analytics Interface:** Built with clean data layout templates and real-time visualization dials driven by Plotly.
* **Cloud Scalability:** Stateless, container-ready microservices architecture designed to handle large-scale test validation requests instantly.


📂 Project Structure
```text
SatyaNetra/
│
├── .gitignore               # Excludes virtual environments and massive model weight binaries
├── requirements.txt         # Production-level dependency list for automated server deployment
├── app.py                   # Main interactive Streamlit application frontend file
├── train.py                 # Core model building, tokenization, and preprocessing engine script
└── model/                   # Local storage container directory for localized model evaluation profiles

**💻 Local Installation & Setup**
To run this repository locally on your machine, follow these steps:

Clone the repository:

git clone [https://github.com/suhanisharma2415-lab/Satya-Netra.git](https://github.com/suhanisharma2415-lab/Satya-Netra.git)
cd Satya-Netra
Set up a virtual environment and activate it:

python -m venv venv
**# On Windows:**
venv\Scripts\activate
**# On Mac/Linux:
source venv/bin/a**ctivate
Install the required production dependencies:

pip install -r requirements.txt
Launch the local development server:

streamlit run app.py
