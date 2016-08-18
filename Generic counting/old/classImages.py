import xml.etree.ElementTree as ET
all_ = {}
class_ = 'bus'
for i in range(1,9963):
    tree = ET.parse('/home/stahl/Annotations/%s.xml'%((format(i, "06d"))))
    root = tree.getroot()
    objects = []
    for o in root.iter('object'):
        obj = o.find('name').text
        if obj not in objects:
            objects.append(obj)
    for obj in objects:
        if obj in all_:
            all_[obj].append(i)
        else:
            all_[obj] = [i]

for class_ in all_:
    f = open('/home/stahl/Generic counting/IO/ClassImages/%s.txt'%(class_),'w+')
    for i in all_[class_]:
        f.write('%i\n'%(i))
        
