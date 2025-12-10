import streamlit as st

st.title("Einfacher Test")
st.write("Wenn Sie das sehen, funktioniert Streamlit!")

if st.button("Test Button"):
    st.success("Button funktioniert!") 