import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import time

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Satya-Netra | AI Fake News Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================
# LOAD MODEL
# ==========================

@st.cache_resource
def load_models():
    model = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

# ==========================
# SESSION STORAGE
# ==========================

if "history" not in st.session_state:
    st.session_state.history = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "Detector"

# ==========================
# ULTRA MODERN CSS
# ==========================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    * {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background: #0a0a0f;
        background-image: 
            radial-gradient(ellipse at 20% 50%, rgba(120, 50, 255, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(0, 200, 255, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(255, 50, 120, 0.05) 0%, transparent 50%);
    }
    
    .glass-card {
        background: rgba(20, 20, 35, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        border: 1px solid rgba(120, 50, 255, 0.3);
        box-shadow: 0 12px 48px rgba(120, 50, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #7832ff, #00c8ff);
        color: white;
        border: none;
        padding: 1rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 16px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 2px;
        width: 100%;
        position: relative;
        overflow: hidden;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 32px rgba(120, 50, 255, 0.4);
    }
    
    .stTextArea textarea {
        background: rgba(15, 15, 25, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        color: #e0e0ff;
        font-size: 1rem;
        padding: 1.2rem;
        line-height: 1.6;
    }
    
    .stTextArea textarea:focus {
        border-color: #7832ff;
        box-shadow: 0 0 30px rgba(120, 50, 255, 0.2);
    }
    
    .stFileUploader > div {
        border: 2px dashed rgba(120, 50, 255, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        background: rgba(120, 50, 255, 0.03);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #7832ff;
        background: rgba(120, 50, 255, 0.06);
    }
    
    .metric-box {
        background: rgba(20, 20, 35, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        border-color: rgba(120, 50, 255, 0.4);
        transform: translateY(-4px);
    }
    
    .metric-number {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #7832ff, #00c8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.5rem;
    }
    
    .result-card-real {
        background: linear-gradient(135deg, rgba(0, 255, 135, 0.1), rgba(0, 200, 255, 0.1));
        border: 1px solid rgba(0, 255, 135, 0.3);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: glowPulse 2s ease-in-out infinite;
    }
    
    .result-card-fake {
        background: linear-gradient(135deg, rgba(255, 50, 50, 0.1), rgba(255, 100, 50, 0.1));
        border: 1px solid rgba(255, 50, 50, 0.3);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: glowPulse 2s ease-in-out infinite;
    }
    
    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 20px rgba(120, 50, 255, 0.1); }
        50% { box-shadow: 0 0 40px rgba(120, 50, 255, 0.2); }
    }
    
    .dataframe {
        background: rgba(20, 20, 35, 0.6) !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .dataframe th {
        background: rgba(120, 50, 255, 0.3) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        border: none !important;
    }
    
    .dataframe td {
        padding: 1rem !important;
        color: rgba(255, 255, 255, 0.8) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(20, 20, 35, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: white;
    }
    
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a0f;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #7832ff, #00c8ff);
        border-radius: 4px;
    }
    
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 16px !important;
        border: none !important;
    }
    
    .stSuccess {
        background: rgba(0, 255, 135, 0.1) !important;
        border: 1px solid rgba(0, 255, 135, 0.3) !important;
        color: #00ff87 !important;
    }
    
    .stError {
        background: rgba(255, 50, 50, 0.1) !important;
        border: 1px solid rgba(255, 50, 50, 0.3) !important;
        color: #ff3232 !important;
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.5px;
    }
    
    .hero-container {
        position: relative;
        text-align: center;
        padding: 3rem 0;
    }
    
    .hero-title {
        font-size: 5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #7832ff 0%, #00c8ff 50%, #ff3278 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: titleGlow 3s ease-in-out infinite;
        letter-spacing: -2px;
    }
    
    @keyframes titleGlow {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.3); }
    }
    
    .hero-subtitle {
        color: rgba(255, 255, 255, 0.6);
        font-size: 1.2rem;
        letter-spacing: 4px;
        text-transform: uppercase;
    }
    
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #7832ff, #00c8ff, #7832ff, transparent);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# NAVIGATION BAR
# ==========================

st.markdown("""
<div style="text-align: center; padding: 1rem 0; margin-bottom: 2rem;">
    <div style="display: inline-block; background: rgba(20, 20, 35, 0.5); 
                backdrop-filter: blur(20px); border-radius: 50px; padding: 0.5rem; 
                border: 1px solid rgba(255, 255, 255, 0.1);">
""", unsafe_allow_html=True)

col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)

with col_nav1:
    if st.button("🔍 Detector", key="nav_detector", use_container_width=True):
        st.session_state.current_page = "Detector"

with col_nav2:
    if st.button("📊 Dashboard", key="nav_dashboard", use_container_width=True):
        st.session_state.current_page = "Dashboard"

with col_nav3:
    if st.button("📜 History", key="nav_history", use_container_width=True):
        st.session_state.current_page = "History"

with col_nav4:
    if st.button("ℹ️ About", key="nav_about", use_container_width=True):
        st.session_state.current_page = "About"

st.markdown("</div></div>", unsafe_allow_html=True)

# ==========================
# DETECTOR PAGE
# ==========================

if st.session_state.current_page == "Detector":
    
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">SATYA-NETRA</div>
        <div class="hero-subtitle">AI-Powered Truth Detection</div>
        <div class="gradient-divider"></div>
    </div>
    """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        st.markdown("""
        <h3 style="color: white; margin-bottom: 1.5rem;">
            <span style="background: linear-gradient(135deg, #7832ff, #00c8ff); 
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                📝 Input News Article
            </span>
        </h3>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "📁 Upload a .txt file",
            type=["txt"],
            help="Supports .txt files"
        )
        
        article_text = ""
        
        if uploaded_file:
            article_text = uploaded_file.read().decode("utf-8")
            st.success("✅ File loaded successfully")
        
        news = st.text_area(
            "Paste your news article below",
            value=article_text,
            height=250,
            placeholder="Paste the news article you want to verify..."
        )
        
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            analyze_btn = st.button("⚡ ANALYZE NOW", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        st.markdown("""
        <h3 style="text-align: center; color: white; margin-bottom: 1.5rem;">
            <span style="background: linear-gradient(135deg, #7832ff, #00c8ff); 
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🎯 Analysis Result
            </span>
        </h3>
        """, unsafe_allow_html=True)
        
        if analyze_btn:
            if news.strip() == "":
                st.warning("⚠️ Please enter some text to analyze")
            else:
                with st.spinner("🔄 AI is analyzing your news..."):
                    time.sleep(0.5)
                    
                    vector = vectorizer.transform([news])
                    prediction = model.predict(vector)[0]
                    proba = model.predict_proba(vector)[0]
                    confidence = proba[1] if prediction == 1 else proba[0]
                    confidence_percent = round(confidence * 100, 2)
                    
                    result = "REAL NEWS ✅" if prediction == 1 else "FAKE NEWS ❌"
                    
                    st.session_state.history.append({
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Result": result,
                        "Confidence": f"{confidence_percent}%",
                        "Preview": news[:150] + "..."
                    })
                    
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="result-card-real">
                            <div style="font-size: 3rem; margin-bottom: 0.5rem;">✅</div>
                            <h2 style="color: #00ff87; margin: 0.5rem 0;">REAL NEWS</h2>
                            <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem;">
                                Confidence Score: <strong>{confidence_percent}%</strong>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card-fake">
                            <div style="font-size: 3rem; margin-bottom: 0.5rem;">❌</div>
                            <h2 style="color: #ff3232; margin: 0.5rem 0;">FAKE NEWS</h2>
                            <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem;">
                                Confidence Score: <strong>{confidence_percent}%</strong>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence_percent,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Confidence Level", 'font': {'size': 20, 'color': '#e0e0ff'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': "#e0e0ff", 'tickfont': {'color': '#e0e0ff'}},
                            'bar': {'color': "rgba(120, 50, 255, 0.8)", 'thickness': 0.2},
                            'bgcolor': "rgba(255,255,255,0.05)",
                            'borderwidth': 0,
                            'steps': [
                                {'range': [0, 33], 'color': 'rgba(255, 50, 50, 0.3)'},
                                {'range': [33, 66], 'color': 'rgba(255, 200, 0, 0.3)'},
                                {'range': [66, 100], 'color': 'rgba(0, 255, 135, 0.3)'}
                            ],
                        }
                    ))
                    
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': "#e0e0ff", 'family': "Space Grotesk"},
                        height=280,
                        margin=dict(l=30, r=30, t=60, b=30)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 0;">
                <div style="font-size: 5rem; margin-bottom: 1rem; opacity: 0.3;">🤖</div>
                <p style="color: rgba(255,255,255,0.4); font-size: 1.1rem;">AI Ready</p>
                <p style="color: rgba(255,255,255,0.3); font-size: 0.9rem;">
                    Paste news and click Analyze
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# DASHBOARD
# ==========================

elif st.session_state.current_page == "Dashboard":
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="background: linear-gradient(135deg, #7832ff, #00c8ff); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 2.5rem; font-weight: 700;">
            Analytics Dashboard
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    total = len(st.session_state.history)
    real_count = sum(1 for x in st.session_state.history if "REAL" in x["Result"])
    fake_count = sum(1 for x in st.session_state.history if "FAKE" in x["Result"])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-number">{total}</div>
            <div class="metric-label">Total Scans</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-number" style="background: linear-gradient(135deg, #00ff87, #00c8ff); 
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {real_count}
            </div>
            <div class="metric-label">Real News</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-number" style="background: linear-gradient(135deg, #ff3232, #ff7846); 
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {fake_count}
            </div>
            <div class="metric-label">Fake News</div>
        </div>
        """, unsafe_allow_html=True)
    
    if total > 0:
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            fig = go.Figure(data=[go.Pie(
                labels=['Real', 'Fake'],
                values=[real_count, fake_count],
                hole=.5,
                marker_colors=['#00ff87', '#ff3232'],
                textinfo='label+percent',
                textfont=dict(color='white', size=14, family='Space Grotesk'),
                hoverinfo='label+value'
            )])
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                height=400,
                margin=dict(t=30, b=30)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_chart2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=['Real News', 'Fake News'],
                    y=[real_count, fake_count],
                    marker=dict(
                        color=['#00ff87', '#ff3232'],
                        line=dict(color='rgba(255,255,255,0.2)', width=1),
                        cornerradius=15
                    ),
                    text=[real_count, fake_count],
                    textposition='outside',
                    textfont=dict(color='white', size=18, family='Space Grotesk'),
                    width=0.4
                )
            ])
            
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, tickfont=dict(color='white', size=14)),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='white')),
                height=400,
                margin=dict(t=30, b=30)
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 4rem;">
            <div style="font-size: 5rem; opacity: 0.2;">📊</div>
            <h3 style="color: rgba(255,255,255,0.5);">No Data Yet</h3>
            <p style="color: rgba(255,255,255,0.3);">Analyze some news articles first</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================
# HISTORY
# ==========================

elif st.session_state.current_page == "History":
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="background: linear-gradient(135deg, #7832ff, #00c8ff); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 2.5rem; font-weight: 700;">
            Prediction History
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    if len(st.session_state.history) == 0:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 4rem;">
            <div style="font-size: 5rem; opacity: 0.2;">📜</div>
            <h3 style="color: rgba(255,255,255,0.5);">No History</h3>
            <p style="color: rgba(255,255,255,0.3);">Your predictions will appear here</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        
        history_df = pd.DataFrame(st.session_state.history)
        
        def style_df(df):
            def highlight(val):
                if "REAL" in str(val):
                    return 'background: rgba(0, 255, 135, 0.15); color: #00ff87; font-weight: 600; border-radius: 8px; padding: 4px 12px;'
                elif "FAKE" in str(val):
                    return 'background: rgba(255, 50, 50, 0.15); color: #ff3232; font-weight: 600; border-radius: 8px; padding: 4px 12px;'
                return ''
            return df.style.applymap(highlight, subset=['Result'])
        
        styled_df = style_df(history_df)
        st.dataframe(styled_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# ABOUT
# ==========================

elif st.session_state.current_page == "About":
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="background: linear-gradient(135deg, #7832ff, #00c8ff); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 2.5rem; font-weight: 700;">
            About Satya-Netra
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #7832ff; margin-bottom: 1rem;">🚀 Overview</h3>
            <p style="color: rgba(255,255,255,0.8); line-height: 1.8; font-size: 1.05rem;">
                Satya-Netra is a cutting-edge AI system that leverages advanced 
                Natural Language Processing to detect misinformation. Built with 
                state-of-the-art machine learning algorithms, it provides instant 
                verification of news authenticity.
            </p>
            <br>
            <h3 style="color: #7832ff; margin-bottom: 1rem;">✨ Features</h3>
            <ul style="color: rgba(255,255,255,0.8); line-height: 2.2; font-size: 1rem;">
                <li>⚡ Real-time Analysis</li>
                <li>🎯 High Accuracy Detection</li>
                <li>📊 Visual Analytics</li>
                <li>📁 File Upload Support</li>
                <li>📈 Confidence Scoring</li>
                <li>💾 History Tracking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #7832ff; margin-bottom: 1rem;">🛠️ Tech Stack</h3>
            <div style="color: rgba(255,255,255,0.8); line-height: 2.5; font-size: 1.05rem;">
                <div>🐍 <strong>Python</strong> - Core Language</div>
                <div>🎨 <strong>Streamlit</strong> - UI Framework</div>
                <div>🧠 <strong>Scikit-Learn</strong> - ML Engine</div>
                <div>📊 <strong>Plotly</strong> - Visualizations</div>
                <div>📝 <strong>TF-IDF</strong> - Text Vectorization</div>
                <div>🤖 <strong>Logistic Regression</strong> - Model</div>
            </div>
            <br>
            <div style="background: rgba(120, 50, 255, 0.1); border-radius: 16px; 
                        padding: 1.5rem; text-align: center; border: 1px solid rgba(120, 50, 255, 0.2);">
                <div style="font-size: 3rem; font-weight: 700; 
                            background: linear-gradient(135deg, #00ff87, #00c8ff);
                            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    98.5%
                </div>
                <div style="color: rgba(255,255,255,0.6); margin-top: 0.5rem;">
                    Model Accuracy
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================
# FOOTER
# ==========================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <div class="gradient-divider"></div>
    <p style="color: rgba(255,255,255,0.3); font-size: 0.9rem; margin-top: 1rem;">
        🛡️ Satya-Netra AI • Truth Matters • Powered by Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)