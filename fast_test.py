from ultralytics import YOLO

# Eğitilen modeli yükle
model = YOLO("runs/helmet-detection/weights/best.pt")

# Test için bir görüntü kullan (örneğin kendi çektiğin bir motorlu görüntüsü)
results = model("frames/044_frame_0020.jpg", save=True, conf=0.6)

# Tahmin sonucu şu klasöre kaydedilir:
# runs/detect/predict/ornek.jpg


'''
Eğitilmiş modeli hızlıca test etmek için kullanılır.

Genelde tek bir görsel ya da kısa video üzerinde inference yapılır.
'''