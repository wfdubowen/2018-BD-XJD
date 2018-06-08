import cv2
import os

i = 0
for filename in os.listdir(r"./train_improve_v4"):
    image = cv2.imread('./train_improve_v4/'+ str(filename))
    res = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('./traslateImg_v4/'+str(filename), res)
    i = i+1
    print(filename, '已经完成', str(i))

print("Done.")
cv2.destroyAllWindows()



# image=cv2.imread('test.jpg')
# res=cv2.resize(image,(100,100),interpolation=cv2.INTER_CUBIC)
# cv2.imshow('iker',res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()