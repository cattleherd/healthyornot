import gradio as gr
from fastai.learner import load_learner
from fastai.vision.core import PILImage

learn = load_learner('healthyornot.pkl')

def classify_image(input_image):
    img = PILImage.create(input_image)
    prediction = learn.predict(img)
    return prediction[0]

iface = gr.Interface(fn=classify_image, 
                     inputs=gr.inputs.Image(shape=(224, 224)), 
                     outputs="text")
iface.launch()