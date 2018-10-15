import re

# src=\".*?\"
FILE = r'C:\Users\AosChen\Desktop\新建文本文档.html'

F = open(FILE, 'r', encoding='UTF-8')
text = F.read()
pattern = re.compile(r'(src=".*?")')
temp = pattern.findall(text)
text = []
for i in temp:
    if '..180x180.jpg' in i:
        continue
    text.append(i.split('"')[1])

write = open(r'C:\Users\AosChen\Desktop\result.txt', 'w')
for i in text:
    write.write('http:' + i + '\n')