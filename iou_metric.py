def iou_metric_threshold(threshold=0.5):
    def iou(y_true, y_pred):
        y_pred = K.cast(K.greater(y_pred, threshold), dtype='float32')
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(K.maximum(y_true, y_pred), axis=[1, 2, 3])
        iou_score = K.mean((intersection + K.epsilon()) / (union + K.epsilon()))
        return iou_score
    return iou