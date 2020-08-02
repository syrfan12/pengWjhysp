# penggunaan : python3 pengolahan_wajah.py
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
from imutils.video import FPS
import pickle
import time
import cv2
import os
import face_recognition
conf = 0.5

# menyiapkan berkas face detektor
print("[INFO] mengakses berkas face detektor...")
data = pickle.loads(open("encodings.pickle", "rb").read())
protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detector",
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# mengakses model pembeda fota/asli
print("[INFO] mengakses model pembeda fota/asl...")
model = load_model("pemb.model")
le = pickle.loads(open("pemb.pickle", "rb").read())

# memulai webcam
print("[INFO] memulai webcam...")
vs = FileVideoStream("vdUcp.mp4").start()
time.sleep(2.0)
fps = FPS().start()

while True:
	fps.update()
	# membaca data video dan mengubah width resolusinya menjadi 600
	frame = vs.read()

	frame = imutils.rotate(frame, angle=180)
    
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = imutils.resize(frame, width=600)

	# blob detecton untuk menentukan posisi yang mirip wajah di kamera
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pendeteksian wajah
	net.setInput(blob)
	detections = net.forward()

	# perulangan untuk setiap wajah yang terdeteksi
	for i in range(0, detections.shape[2]):
		# mendapatkan nilai probabilitas terdekesinya wajah dari kamera
		confidence = detections[0, 0, i, 2]

		# menghilangkan pendeteksian yang probabilitasnya rendah
		if confidence > conf:
			# menentukan koordinat wajah yang memiliki probabilitas tinggi
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# memastikan agar gambar kotak tidak masuk ke dalam frame yang akan diolah
			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)

			# mendapatkan bagian ROI yang tepat
			face = frame[startY:endY, startX:endX]
			face = cv2.resize(face, (32, 32))
			face = face.astype("float") / 255.0
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)

			# melakukan pengujian pada wajah yang terdeteksi untuk menentukan
			# asli atau foto
			preds = model.predict(face)[0]
			j = np.argmax(preds)
			label = le.classes_[j]

			# memberi gambar kotak pada wajah yang dideteksi dan label apakah
			# foto atau asli
			if label == "foto":
				cv2.putText(frame, "foto: {:.4f}".format(preds[j]), (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
			else:
				# jika asli maka dilakukan proses face recognition untuk mengetahui wajah siapa
				encodings = face_recognition.face_encodings(rgb, [[startY, endX, endY, startX]])
				names = []
				
				for encoding in encodings:
					matches = face_recognition.compare_faces(data["encodings"],
						encoding)
					name = "tidak dikenali"

				# check apakah ada wajah yang di kenali
				if True in matches:
					matchedIdxs = [i for (i, b) in enumerate(matches) if b]
					counts = {}
					for i in matchedIdxs:
						name = data["names"][i]
						counts[name] = counts.get(name, 0) + 1
					name = max(counts, key=counts.get)
				names.append(name)
				# menampilkan nama wajah dan memberi area kotak
				cv2.putText(frame, name, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 200, 0), 2)

	
	fps.stop()
	cv2.putText(frame, "FPS: {:.2f}".format(fps.fps()), (10, 20),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
