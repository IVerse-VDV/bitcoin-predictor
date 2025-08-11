import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from bitcoin_model import BitcoinPredictor, ai_predict, load_model

# page config
st.set_page_config(
    page_title="Bitcoin AI Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed"
)





# CSS
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, .stDeployButton {display: none;}
    

    
    /* header styling */
    .header-section {
        text-align: center;
        margin-bottom: 3rem;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 2rem;
    }

    
    /* input section */
    .input-section {
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        color: white;
    }
    
    .input-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3b82f6, #6366f1, #8b5cf6);
    }
    
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: white;
    }
    
    /* priceinput grid */
    .price-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .price-item {
        background: black;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .price-item:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        transform: translateY(-2px);
    }
    
    .day-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    
    /* prediction section */
    .prediction-container {
        background: white;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        border: 1px solid #e2e8f0;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-up {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-color: #22c55e;
    }
    
    .prediction-down {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-color: #ef4444;
    }
    
    .prediction-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .prediction-text {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .prediction-up .prediction-text {
        color: #15803d;
    }
    
    .prediction-down .prediction-text {
        color: #dc2626;
    }
    
    .confidence-text {
        font-size: 1.3rem;
        font-weight: 500;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    /* metrics section */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    

    
    .metric-card:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        transform: translateY(-3px);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* button styl */
    .predict-button {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .predict-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
    
    /* footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #f1f5f9;
    }
    
    /* animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* responsive */
    @media (max-width: 768px) {
        .main-container {
            padding: 2rem 1.5rem;
            margin: 1rem;
        }
        
        .main-title {
            font-size: 2.5rem;
        }
        
        .price-grid {
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 0.75rem;
        }
    }
</style>
""", unsafe_allow_html=True)







def casual_chart(prices):
    """price chart"""
    days = ['6 days ago', '5 days ago', '4 days ago', '3 days ago', '2 days ago', 'Yesterday', 'Today']
    
    # calculate price changes for colorinh
    changes = [0]
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        changes.append(change)
    
    fig = go.Figure()
    
    # main price line
    fig.add_trace(go.Scatter(
        x=days,
        y=prices,
        mode='lines+markers',
        name='Bitcoin Price',
        line=dict(
            color='#3b82f6',
            width=4,
            shape='spline',
            smoothing=0.3
        ),
        marker=dict(
            size=10,
            color='#3b82f6',
            line=dict(width=3, color='white'),
            symbol='circle'
        ),
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.1)',
        hovertemplate='<b>%{x}</b><br>Price: Rp%{y:,.0f}<br><extra></extra>',
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#3b82f6",
            font_size=12,
        )
    ))
    
    # trend line
    x_numeric = list(range(len(prices)))
    z = np.polyfit(x_numeric, prices, 1)
    trend_line = np.poly1d(z)
    
    fig.add_trace(go.Scatter(
        x=days,
        y=[trend_line(i) for i in x_numeric],
        mode='lines',
        name='Trend',
        line=dict(
            color='rgba(239, 68, 68, 0.6)',
            width=2,
            dash='dot'
        ),
        hoverinfo='skip'
    ))
    
    # layout
    fig.update_layout(
        title=dict(
            text="<b>7-Day Bitcoin Price Analysis</b>",
            x=0.5,
            font=dict(size=20, color="#b5bac2", family='Inter')
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.2)',
            gridwidth=1,
            tickfont=dict(size=11, color='#64748b'),
            showline=True,
            linecolor='#e2e8f0'
        ),
        yaxis=dict(
            title="Price (IDR)",
            title_font=dict(size=12, color='#64748b'),
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.2)',
            gridwidth=1,
            tickformat=',.0f',
            tickfont=dict(size=11, color='#64748b'),
            showline=True,
            linecolor='#e2e8f0'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=50),
        hovermode='x unified'
    )
    
    return fig






def main():
    # container
    st.markdown('<div class="main-container fade-in">', unsafe_allow_html=True)
    
    # header section
    st.markdown("""
    <div class="header-section">
        <h1 class="main-title">₿ Bitcoin AI Predictor</h1>
        <p class="subtitle">Neural network analysis with 7-day market intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # model availability
    model, scaler = load_model()
    if model is None:
        st.error("🚫 **Model Not Found** - Please run training first: `python bitcoin_model.py`")
        st.stop()
    
    # input section
    st.markdown("""
    <div class="section-title">
        Historical Price Data Input
    </div>
    """, unsafe_allow_html=True)
    
    # price input grid
    default_prices = [105000, 106200, 105800, 107100, 109900, 110000, 116800]
    day_labels = ['Day -6', 'Day -5', 'Day -4', 'Day -3', 'Day -2', 'Day -1', 'Today']
    
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    cols = [col1, col2, col3, col4, col5, col6, col7]
    prices = []
    
    for i, (col, label) in enumerate(zip(cols, day_labels)):
        with col:
            st.markdown(f'<div class="day-label">{label}</div>', unsafe_allow_html=True)
            price = st.number_input(
                "",
                min_value=20000,
                max_value=50000000,
                value=default_prices[i],
                step=500,
                key=f"price_{i}",
                label_visibility="collapsed"
            )
            prices.append(price)
            st.markdown(f'<div style="text-align: center; font-size: 0.9rem; font-weight: 500; color: #3b82f6;">Rp {price:,.0f}</div>', unsafe_allow_html=True)
    
    # chart section
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(casual_chart(prices), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # predict button
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        predict_clicked = st.button("**Analyze & Predict**", use_container_width=True, type="primary")
    
    #prediction section
    if predict_clicked:
        with st.spinner("Running AI analysis..."):
            try:
                prediction, probability = ai_predict(prices)
                
                if prediction is not None:
                    # Determine prediction styling
                    if prediction == 1:
                        container_class = "prediction-up"
                        icon = "🟢"
                        text = "PRICE WILL INCREASE"
                        subtext = "BULLISH SIGNAL"
                        confidence = probability * 100
                    else:
                        container_class = "prediction-down"
                        icon = "🔴"
                        text = "PRICE WILL GO DOWN"
                        subtext = "BEARISH SIGNAL"
                        confidence = (1 - probability) * 100
                    
                    # display prediction (with animation
                    prediction_html = f"""
                    <div class="prediction-container {container_class} fade-in">
                        <span class="prediction-icon">{icon}</span>
                        <div class="prediction-text">{text}</div>
                        <div style="font-size: 1rem; font-weight: 600; color: #64748b; margin-bottom: 1rem;">{subtext}</div>
                        <div class="confidence-text">AI Confidence Score: {confidence:.1f}%</div>
                        <div style="background: rgba(255,255,255,0.7); border-radius: 12px; padding: 1.5rem; margin-top: 2rem;">
                            <div style="font-size: 0.9rem; color: #64748b; font-weight: 500;">
                                Prediction based on pattern recognition of 7-day price sequences
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(prediction_html, unsafe_allow_html=True)
                    
                    # metrics
                    st.markdown("### **Market Analysis**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        weekly_change = ((prices[-1] - prices[0]) / prices[0]) * 100
                        change_color = "#22c55e" if weekly_change > 0 else "#ef4444"
                        st.markdown(f"""
                        <div class="metric-card fade-in">
                            <div class="metric-value" style="color: {change_color};">{weekly_change:+.2f}%</div>
                            <div class="metric-label">7-Day Change</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        daily_change = ((prices[-1] - prices[-2]) / prices[-2]) * 100
                        change_color = "#22c55e" if daily_change > 0 else "#ef4444"
                        st.markdown(f"""
                        <div class="metric-card fade-in">
                            <div class="metric-value" style="color: {change_color};">{daily_change:+.2f}%</div>
                            <div class="metric-label">1-Day Change</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        volatility = np.std(prices) / np.mean(prices) * 100
                        vol_color = "#ef4444" if volatility > 5 else "#f59e0b" if volatility > 3 else "#22c55e"
                        st.markdown(f"""
                        <div class="metric-card fade-in">
                            <div class="metric-value" style="color: {vol_color};">{volatility:.2f}%</div>
                            <div class="metric-label">Volatility</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        avg_price = np.mean(prices)
                        st.markdown(f"""
                        <div class="metric-card fade-in">
                            <div class="metric-value" style="color: white;">Rp {avg_price:,.0f}</div>
                            <div class="metric-label">7-Day Average</div>
                        </div>
                        """, unsafe_allow_html=True)

                
                else:
                    st.error("❌ **Prediction Error** - Please verify input data and try again.")
                    
            except Exception as e:
                st.error(f"❌ **System Error**: {str(e)}")
    
    # footer
    st.markdown("""
    <div class="footer">
        <p><strong>Risk Disclaimer:</strong> This tool is for educational and research purposes only. 
        Cryptocurrency investments carry substantial risk of loss.</p>
        <p style="margin-top: 1rem;">Powered by PyTorch Neural Networks</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # close container

if __name__ == "__main__":
    main()