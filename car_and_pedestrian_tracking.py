import cv2

#our image

image_file='car_image.jpg'

#our pre-trained car classifier 

classifier_file='car_detector.xml'

#create opencv image

img=cv2.imread(image_file)

# convert to grayscale(needed for haar cascade)
black_n_white=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



#create car classifier

car_tracker=cv2.CascadeClassifier(classifier_file)

#decter cars

cars=car_tracker.detectMultiScale(black_n_white)

print(cars)

#draw rectangle around the car
car1=cars[0]
(x,y,w,h)=car1
cv2.rectangle(img, (x ,y),(x +w , y+ h), (0 ,0 ,255), 2 )

#display the image with car spotted

cv2.imshow("Ahmed car dectector " , img)

#donot autoclose wait here in the code and listen for for a key press 

cv2.waitKey()
print("code completed")