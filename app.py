import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# title
st.title('Qushlarni klassifikatsiya qiluvchi model')

#rasmni joylash
file = st.file_uploader('Rasm Yuklash', type=['png', 'jpg', 'gif'])
if file:
    st.image(file)

    #PIL convert
    img = PILImage.create(file)

    #model
    model = load_learner('birds_model.pkl')

    #prediction
    pred, pred_id, probs = model.predict(img)
    st.success(pred)
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}%')

    # plotting
    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)