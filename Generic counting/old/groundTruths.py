import xml.etree.ElementTree as ET
for i in range(1,9963):
    tree = ET.parse('/home/stahl/Annotations/%s.xml'%((format(i, "06d"))))
    root = tree.getroot()
    objects = {}
    for o in root.iter('object'):
        obj = o.find('name').text
        x1 = int(o.find('bndbox')[0].text)
        y1 = int(o.find('bndbox')[1].text)
        x2 = int(o.find('bndbox')[2].text)
        y2 = int(o.find('bndbox')[3].text)
        if obj not in objects:
            objects[obj] = [[x1,y1,x2,y2]]
        else:
            objects[obj].append([x1,y1,x2,y2])
    for obj in objects:
        f = open('/home/stahl/GroundTruth/%s/%s.txt'%(obj,(format(i, "06d"))),'w+')
        for coords in objects[obj]:
            f.write('%s,%s,%s,%s\n'%(str(coords[0]-1),str(coords[1]-1),str(coords[2]-1),str(coords[3]-1))) #-1 because matlab -> python
#            print '%s,%s,%s,%s'%(str(coords[0]-1),str(coords[1]-1),str(coords[2]-1),str(coords[3]-1))