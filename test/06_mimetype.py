from mimetypes import guess_type

from mpmath.libmp.gammazeta import f3

f1 = 'xx.png'
f2 = 'xx.xlxs'
f3 = 'xx.jpg'
f4 = 'xx.docx'
f5 = 'xx.pdf'
f6 = 'xx.zip'

print(guess_type(f1))
print(guess_type(f2))
print(guess_type(f3))
print(guess_type(f4))
print(guess_type(f5))
print(guess_type(f6))
