import os
import glob
from PIL import Image

files = glob.glob('ここに元画像のディレクトリ名/*.jpg')
a = 0
for f in files:
    a = a+1
    img = Image.open(f)
    img_resize = img.resize((70, 70))　#画像のサイズの指定
    ftitle, fext = os.path.splitext(f)
    img_resize.save('ここにリサイズ後の保存先/' + str(a) + '_(70)' + fext)
