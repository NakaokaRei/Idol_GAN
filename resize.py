import os
import glob
from PIL import Image

files = glob.glob('C:/Users/kimoto/Desktop/new/kurropu/*.jpg')
a = 0
for f in files:
    a = a+1
    img = Image.open(f)
    img_resize = img.resize((70, 70))
    ftitle, fext = os.path.splitext(f)
    img_resize.save('C:/Users/kimoto/Desktop/new/resize/' + str(a) + '_(70)' + fext)
