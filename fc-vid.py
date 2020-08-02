# penggunaan : python3 fc-vid.py

from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# load pendeteksi wajah dari file cascade OpenCV
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open("encodings.pickle", "rb").read())
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Nyalakan Kamera
print("[INFO] Memulai Stream dari Pi Camera...")
vs = VideoStream(src=1).start()

# Penghitung FPS
fps = FPS().start()

# loop dari semua frame yang di dapat
while True:
	# dapatkan frame, dan resize ke 500pixel agar lebih cepat
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	
	# Konversi ke grayscale dan konversi ke RGB
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# deteksi wajah dari frame grayscale
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)
	#print("rects ", rects, type(rects))
   # Tampilkan kotak di wajah yang dideteksi
	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
	for d in range(len(boxes)):
                        boxes[d] = list(map(int, boxes[d]))

	encodings = face_recognition.face_encodings(rgb, boxes)
	print("boxes ", boxes, type(boxes))
	names = []

	# loop di semua wajah yang terdeteksi
	for encoding in encodings:
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Tidak dikenali"

		# check apakah ada wajah yang di kenali
		if True in matches:
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
			name = max(counts, key=counts.get)
		names.append(name)

	# loop di semua wajah yang sudah di kenali
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# tampilkan nama di wajah yang di kenali
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	# Tampilkan gambar di layar
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# tunggu tombol 1 untuk keluar
	if key == ord("q"):
		break

	# update FPS
	fps.update()
	fps.stop()
	#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
cv2.destroyAllWindows()
vs.stop()
