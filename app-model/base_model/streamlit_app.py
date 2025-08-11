import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from bitcoin_model import BitcoinPredictor, predict_price_movement

# Page configuration
st.set_page_config(
    page_title="Bitcoin Price Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #f7931a;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .prediction-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .up-prediction {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .down-prediction {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
    }
    
    .metric-container {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .price-input {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)







def load_model_safe():
    try:
        model = BitcoinPredictor(input_size=3)
        model.load_state_dict(torch.load('bitcoin_model.pth', map_location='cpu'))
        model.eval()
        scaler = joblib.load('scaler.pkl')
        return model, scaler, True
    except Exception as e:
        return None, None, False




def create_price_chart(day1, day2, day3):
    days = ['Day -2', 'Day -1', 'Today']
    prices = [day1, day2, day3]
    
    # calculate price changes
    changes = [0]  # first day has no change reference
    for i in range(1, len(prices)):
        change = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
        changes.append(change)
    
    # figure
    fig = go.Figure()
    
    #add price line
    fig.add_trace(go.Scatter(
        x=days,
        y=prices,
        mode='lines+markers',
        name='Bitcoin Price',
        line=dict(color='#f7931a', width=4),
        marker=dict(size=12, color='#f7931a', line=dict(width=2, color='white')),
        hovertemplate='<b>%{x}</b><br>Price: Rp%{y:,.0f}<extra></extra>'
    ))
    
    # layout
    fig.update_layout(
        title=dict(
            text="Bitcoin Price Trend (Last 3 Days)",
            x=0.5,
            font=dict(size=20, color='#2c3e50')
        ),
        xaxis=dict(
            title="Days",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            title="Price (IDR)",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            tickformat=',.0f'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        showlegend=False
    )
    
    return fig




def main():
    # header
    st.markdown('<h1 class="main-header">₿ Bitcoin Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d; margin-bottom: 3rem;">Predict tomorrow\'s Bitcoin price movement using AI</p>', unsafe_allow_html=True)
    
    # check if model exists
    model, scaler, model_loaded = load_model_safe()
    
    if not model_loaded:
        st.error("Model not found! Please run `python bitcoin_model.py` first to train the model.")
        st.info("The training process will create the necessary model files.")
        st.stop()
    
    # Create input section
    st.markdown("## Enter Bitcoin Prices (Last 3 Days)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Day -2")
        day1 = st.slider(
            "Price (IDR)",
            min_value=20000,
            max_value=100000,
            value=50000,
            step=1000,
            key="day1",
            help="Bitcoin price 2 days ago"
        )
        st.markdown(f'<p class="price-input">Rp {day1:,.0f}</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Day -1")
        day2 = st.slider(
            "Price (IDR)",
            min_value=20000,
            max_value=100000,
            value=51000,
            step=1000,
            key="day2",
            help="Bitcoin price 1 day ago"
        )
        st.markdown(f'<p class="price-input">Rp {day2:,.0f}</p>', unsafe_allow_html=True)
    
    with col3:
        st.markdown("### Today")
        day3 = st.slider(
            "Price (IDR)",
            min_value=20000,
            max_value=100000,
            value=52000,
            step=1000,
            key="day3",
            help="Bitcoin price today"
        )
        st.markdown(f'<p class="price-input">Rp {day3:,.0f}</p>', unsafe_allow_html=True)
    
    # create price chart
    st.plotly_chart(create_price_chart(day1, day2, day3), use_container_width=True)
    
    # make prediction
    with st.spinner("Analyzing price patterns..."):
        prediction, probability = predict_price_movement(day1, day2, day3)
    
    if prediction is not None:
        # Display prediction
        st.markdown("---")
        st.markdown("## Prediction Result")
        
        # Determine prediction class and styling
        if prediction == 1:
            pred_class = "up-prediction"
            pred_icon = "🟢"
            pred_text = "INCREASE"
            pred_emoji = "🚀"
        else:
            pred_class = "down-prediction"
            pred_icon = "🔴"
            pred_text = "DOWN"
            pred_emoji = "📉"
        
        # prediction display
        percentage = probability * 100 if prediction == 1 else (1 - probability) * 100
        
        prediction_html = f"""
        <div class="prediction-container {pred_class}">
            <div style="text-align: center;">
                <h2 style="margin: 0; font-size: 2.5rem;">{pred_emoji} PRICE tomorrow is {pred_text}</h2>
                <h3 style="margin: 1rem 0; font-size: 2rem;">Confidence: {percentage:.1f}%</h3>
            </div>
            <div class="metric-container" style="margin-top: 2rem;">
                <div style="display: flex; justify-content: space-around; align-items: center;">
                    <div style="text-align: center;">
                        <h4 style="margin: 0; color: rgba(255,255,255,0.8);">Current Trend</h4>
                        <p style="margin: 0.5rem 0; font-size: 1.5rem;">{pred_icon}</p>
                    </div>
                    <div style="text-align: center;">
                        <h4 style="margin: 0; color: rgba(255,255,255,0.8);">AI Confidence</h4>
                        <p style="margin: 0.5rem 0; font-size: 1.5rem;">{percentage:.1f}%</p>
                    </div>
                    <div style="text-align: center;">
                        <h4 style="margin: 0; color: rgba(255,255,255,0.8);">Risk Level</h4>
                        <p style="margin: 0.5rem 0; font-size: 1.5rem;">{"Low" if percentage > 70 else "Medium" if percentage > 60 else "High"}</p>
                    </div>
                </div>
            </div>
        </div>
        """
        
        st.markdown(prediction_html, unsafe_allow_html=True)
        
        # additional metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_change_2d = ((day3 - day1) / day1) * 100
            st.metric("2-Day Change", f"{price_change_2d:+.2f}%", f"{day3 - day1:+,.0f}")
        
        with col2:
            price_change_1d = ((day3 - day2) / day2) * 100
            st.metric("1-Day Change", f"{price_change_1d:+.2f}%", f"{day3 - day2:+,.0f}")
        
        with col3:
            volatility = np.std([day1, day2, day3]) / np.mean([day1, day2, day3]) * 100
            st.metric("Volatility", f"{volatility:.2f}%")
        
        with col4:
            avg_price = np.mean([day1, day2, day3])
            st.metric("3-Day Average", f"Rp {avg_price:,.0f}")
    
    else:
        st.error("❌ Prediction failed. Please check the model files.")
    
    # footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 2rem;">
        <p><strong>⚠️ Disclaimer:</strong> This is for educational purposes only. Cryptocurrency trading involves high risk.</p>
        <p>Powered by PyTorch Neural Network</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()