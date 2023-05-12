import streamlit as st
from fastai.vision.all import *
import gdown
from PIL import Image
import altair

st.markdown("""# Gender Classifier Galbadral

As I have done gender calssification from the black white picture dataframe with different models

This time I tried to show how fastai is compare to them

This app was created as a demo for the Deep Learning course at LETU Mongolia American University.""")


st.markdown("""### Upload your image here""")

image_file = st.file_uploader("Image Uploader", type=["png","jpg","jpeg"])

## Model Loading Section
model_path = Path("export.pkl")

if not model_path.exists():
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        url = 'https://drive.google.com/uc?id=1YCPTZAjsf13-tECx3wX2uX_OQntBTyHz'
        output = 'export.pkl'
        gdown.download(url, output, quiet=False)
    learn_inf = load_learner('export.pkl')
else:
    learn_inf = load_learner('export.pkl')

col1, col2 = st.columns(2)
if image_file is not None:
    img = PILImage.create(image_file)
    pred, pred_idx, probs = learn_inf.predict(img)

    with col1:
        st.markdown(f"""### Predicted food: {pred.capitalize()}""")
        st.markdown(f"""### Probability: {round(max(probs.tolist()), 3) * 100}%""")
    with col2:
        st.image(img, width=300)
        