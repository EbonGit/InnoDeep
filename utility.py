def progress_bar(percent):
    bar_width = 60
    filled_width = int(bar_width * percent)
    bar = '█' * filled_width + '-' * (bar_width - filled_width)
    return (f'Normal → [{bar}] ← Cancer \n{round(percent*100, 2)}%')


import cv2
import numpy as np


def extract_rois_from_mask(mask):
    # Convertir le masque en image binaire à canal unique de 8 bits
    _, binary_mask = cv2.threshold(mask, 32, 255, cv2.THRESH_BINARY)

    # Trouver les contours des zones d'intérêt dans le masque binaire
    contours, hierarchies = cv2.findContours(image=binary_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    rois = []
    boxs = []

    # Extraire et rogner chaque zone d'intérêt
    for contour in contours:
        # Récupérer les coordonnées du rectangle englobant (bounding box)
        x, y, w, h = cv2.boundingRect(contour)

        # Appliquer un zoom sur la zone d'intérêt
        zoom_factor = 1.2
        new_x = int(x - (zoom_factor - 1) * w / 2)
        new_y = int(y - (zoom_factor - 1) * h / 2)
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)

        # Limiter les coordonnées pour éviter les sorties hors de l'image
        new_x = max(new_x, 0)
        new_y = max(new_y, 0)
        new_w = min(new_w, mask.shape[1])
        new_h = min(new_h, mask.shape[0])

        new_w = new_h = max(new_h, new_w)

        # Extraire la zone d'intérêt du masque
        roi = mask[new_y:new_y + new_h, new_x:new_x + new_w]

        # Ajouter la zone d'intérêt à la liste des ROIs
        rois.append(roi)
        boxs.append([new_x, new_y, new_w, new_h])

    return rois, boxs


def fade_green_to_red(percentage):
    if percentage < 0 or percentage > 100:
        raise ValueError("Le pourcentage doit être compris entre 0 et 100.")

    green = (1 * (100 - percentage) / 100)
    red = (1 * percentage / 100)

    return red, green, 0


def afficher_image_avec_zone_encadree(image, x, y, w, h, text, color_val, delta_y=0):
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir l'image de BGR à RGB
    image_rgb = image

    color = fade_green_to_red(color_val)

    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), color,
                  int(image_rgb.shape[0] / 400))  # Couleur verte (0, 255, 0) et épaisseur 2 pixels

    text_position = (x + w + 10, (y + h // 2) + delta_y)  # Position du texte à côté de la zone encadrée
    cv2.putText(image_rgb, text, text_position, cv2.FONT_HERSHEY_DUPLEX, image_rgb.shape[0] / 1200, color,
                int(image_rgb.shape[0] / 400))  # Couleur verte (0, 255, 0) et épaisseur 2 pixels

    return image_rgb