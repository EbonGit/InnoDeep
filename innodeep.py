from img_process import *
from utility import *
from iou_metric import iou_metric_threshold
import streamlit as st
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from io import BytesIO


def upload_to_cv2(upload):
    pil_img = Image.open(upload)
    img = np.array(pil_img)
    st.write(img.shape)
    return img


model_path = "model/model.h5"
model_inverse_path = "model/model_inverse.h5"
model_classification_path = "model/model_classification.h5"
model_ROI_path = "model/model_ROI.h5"
model_ROI_type_path = "model/model_ROI_type.h5"


class TumorSeeker:
    def __init__(self, model_path=None, model_inverse_path=None, model_classification_path=None, model_ROI_path=None,
                 model_ROI_type_path=None):
        self.model = None
        self.model_inverse = None
        self.model_classification = None
        self.model_ROI = None
        self.model_ROI_type = None

        self.data_raw = []
        self.data_fullsize = []
        self.data = np.zeros((128, 128))
        self.prediction_fullsize = []
        self.prediction_inverse_fullsize = []
        self.prediction = np.zeros((128, 128))
        self.prediction_inverse = np.zeros((128, 128))

        self.prediction_box = []
        self.rois = []

        self.predicition_label = 1

        self.pred_classification = -1

        if model_path:
            if model_inverse_path:
                self.load_model(model_path, model_inverse_path)
            else:
                self.load_model(model_path)
        else:
            print("you need to initialize the model")
        if model_classification_path:
            self.load_classification(model_classification_path)
        if model_ROI_path:
            if model_ROI_type_path:
                self.load_ROI(model_ROI_path, model_ROI_type_path)
            else:
                self.load_ROI(model_ROI_path)

    def load_data(self, data_path):
        self.data_raw = upload_to_cv2(data_path)
        print("data_raw shape:         ", self.data_raw.shape)

    def preprocess_data(self):
        # Implement the logic for data preprocessing
        self.data, _ = preprocess_pipeline(self.data_raw)
        print("preprocess_data shape : ", self.data.shape)
        self.data_fullsize = self.data.copy()
        self.data = cv2.resize(self.data, (128, 128))
        print("resize_data shape :      ", self.data.shape)
        self.data = np.expand_dims(self.data, 0)

    def predict(self):
        # Implement the logic to make predictions using the trained tumor detection model
        self.preprocess_data()
        self.prediction = np.squeeze(self.model.predict(self.data, verbose=0), axis=0)
        self.prediction_fullsize = cv2.resize(self.prediction, (self.data_fullsize.shape))

        if self.model_inverse:
            self.prediction_inverse = np.squeeze(self.model_inverse.predict(self.data, verbose=0), axis=0)
            self.prediction_fullsize_inverse = cv2.resize(self.prediction_inverse, (self.data_fullsize.shape))

    def predict_label(self):
        pred = self.model_classification.predict(np.expand_dims(self.data, axis=-1), verbose=0)[0][0]
        self.predicition_label = pred
        pred = progress_bar(pred)
        return str(pred)

    def save_model(self, model_path):
        # Implement the logic to save the trained tumor detection model
        self.model.save(model_path)

    def load_model(self, model_path, model_inverse_path=None):
        # Implement the logic to load a saved tumor detection model
        self.model = keras.models.load_model(model_path, custom_objects={
            'binary_cross_entropy': tf.keras.losses.BinaryCrossentropy(from_logits=False),
            'iou': iou_metric_threshold(0.2)})
        if model_inverse_path:
            self.model_inverse = keras.models.load_model(model_inverse_path, custom_objects={
                'binary_cross_entropy': tf.keras.losses.BinaryCrossentropy(from_logits=False),
                'iou': iou_metric_threshold(0.2)})

    def load_classification(self, model_classification_path):
        self.model_classification = keras.models.load_model(model_classification_path, custom_objects={
            'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy()})

    def load_ROI(self, model_ROI_path, model_ROI_type_path=None):
        self.model_ROI = keras.models.load_model(model_ROI_path, custom_objects={
            'binary_crossentropy': tf.keras.losses.BinaryCrossentropy()})
        if model_ROI_type_path:
            self.model_ROI_type = keras.models.load_model(model_ROI_type_path, custom_objects={
                'binary_crossentropy': tf.keras.losses.BinaryCrossentropy()})

    def process_bounding_box(self, coef):
        mask = np.array(self.prediction * 255, np.uint8)
        rois, boxs = extract_rois_from_mask(mask)

        img_fullsize = self.data_fullsize.copy()
        img_fullsize = cv2.merge((img_fullsize, img_fullsize, img_fullsize))
        ratio = img_fullsize.shape[0] / self.prediction.shape[0]

        self.rois = []

        for box in boxs:
            box = [round(i * ratio) for i in box]
            self.rois.append(cv2.resize(
                np.repeat(self.data_fullsize[box[1]:box[1] + box[3], box[0]:box[0] + box[2]][..., np.newaxis], 3,
                          axis=2), (128, 128)))

            tumour_val = 0
            type_val = 0
            if self.model_ROI != None:
                tumour_val = self.model_ROI.predict(np.expand_dims(self.rois[-1] * 255, axis=0), verbose=0)[0][0] * coef
                tumour_val = int(tumour_val * 1000) / 1000

            if self.model_ROI_type != None:
                type_val = self.model_ROI_type.predict(np.expand_dims(self.rois[-1] * 255, axis=0), verbose=0)[0][0]

            img_fullsize = afficher_image_avec_zone_encadree(img_fullsize, box[0], box[1], box[2], box[3],
                                                             "TUMOUR " + str(tumour_val), int(tumour_val * 100), -100)

            if type_val >= 0.5:
                type_val = int(type_val * coef * 1000) / 1000
                img_fullsize = afficher_image_avec_zone_encadree(img_fullsize, box[0], box[1], box[2], box[3],
                                                                 "CALC  " + str(type_val), int(tumour_val * 100), 0)
            else:
                type_val = int((1 - type_val) * coef * 1000) / 1000
                img_fullsize = afficher_image_avec_zone_encadree(img_fullsize, box[0], box[1], box[2], box[3],
                                                                 "MASS  " + str(type_val), int(tumour_val * 100), 0)

        self.prediction_box = img_fullsize

    def show(self, option="normal"):
        if self.model_inverse:
            fig, ax = plt.subplots(2, 3, figsize=(10, 5))
            mult = np.multiply(self.prediction_fullsize, 1 - self.prediction_fullsize_inverse)
            ax[1][0].imshow(mult)
            ax[1][0].set_title('Multiply')
        else:
            fig, ax = plt.subplots(2, 2, figsize=(20, 10))

        fig.tight_layout(pad=1.0)
        ax[0][0].imshow(self.data_fullsize, cmap='gray')
        ax[0][0].set_title('Data')

        ax[1][1].imshow(np.squeeze(self.data_fullsize), cmap='gray')
        ax[1][1].imshow(mult, alpha=0.4, cmap=None)

        ax[0][2].imshow(1 - self.prediction_fullsize_inverse, alpha=1, cmap=None)
        if option == "treshold":
            _, tresh = cv2.threshold(self.prediction_fullsize, 0.9, 1, cv2.THRESH_BINARY)
            ax[0][1].imshow(tresh, alpha=1, cmap=None)
        if option == "normal":
            ax[0][1].imshow(self.prediction_fullsize, alpha=1, cmap=None)
        ax[0][1].set_title('Prediction')
        ax[0][2].set_title('Prediction Inverse')

        ax[1][1].set_title('Mask')
        ax[1][2].set_title('Bounding Box')

        pred_label = self.predict_label()

        self.process_bounding_box(self.predicition_label)

        ax[1][2].imshow(self.prediction_box, cmap=None)

        # fig.suptitle(self.predict_label(), fontsize=16)
        st.text(pred_label)
        st.pyplot(fig)

@st.cache_resource(max_entries=1)
def load_models():
    seeker = TumorSeeker(model_path, model_inverse_path, model_classification_path, model_ROI_path, model_ROI_type_path)
    return seeker


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

seeker = load_models()

st.title("-INNODEEP- TUMOUR DETECTION")

img_upload = st.file_uploader("select", type=['png', 'jpg', 'jpeg'])

if img_upload:
    with st.spinner('Wait for it...'):
        seeker.load_data(img_upload)
        seeker.predict()
        seeker.show()

    st.image(seeker.prediction_box)

