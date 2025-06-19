import streamlit as st

# ✅ CSS Styling
st.markdown("""
    <style>
        .stApp {
            background-color: #3470ac;
            min-height: 100vh;
            padding-top: 60px;
        }

        .scoliosis-text {
            font-weight: bold;
            color: black;
            font-size: 60px;
            text-align: center;
            margin-top: 150px;
            margin-bottom: 40px;
        }

        .click-here-button {
            display: flex;
            justify-content: flex-end;
            margin-right: 180px; /* 🔵 เยื้องขวาเยอะขึ้น */
            margin-top: 20px;
        }

        .click-here-button > button {
            font-size: 36px;
            font-weight: bold;
            padding: 20px 60px;
            background-color: white;
            color: black;
            border: 5px solid yellow;
            border-radius: 12px;
            box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.3);
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ ข้อความ SCOLIOSIS ตรงกลาง
st.markdown('<div class="scoliosis-text">SCOLIOSIS</div>', unsafe_allow_html=True)

# ✅ ปุ่ม CLICK HERE เยื้องขวา
with st.container():
    st.markdown('<div class="click-here-button">', unsafe_allow_html=True)
    if st.button("CLICK HERE"):
        st.switch_page("page1.py")  # ✅ correct!
    st.markdown('</div>', unsafe_allow_html=True)
