"""
在数据标注完成后，制作成VOC2007格式数据，
并转化为yolo3，能使用的数据，并保存在文件夹下

"""
import os
import cfg
import xml.etree.ElementTree as ET


image_saved_dir = "./data/train_data/quick_train_data/"
txt_save_file = "./data/train_data/train_data.txt"


def convert_annotation(xml_dir, txt_save_file, classes_names_file=cfg.classes_names_file):
    # create class dictionary
    classlist = []
    with open(classes_names_file, "r", encoding="utf-8") as f:
        for line in f:
            classlist.append(line.strip())
    classes_dict = {name: i for (i, name) in enumerate(classlist, start=0)}

    xml_list = os.listdir(xml_dir)
    with open(txt_save_file, "w", encoding="utf-8") as f:
        count = 0
        for xml_file in xml_list:
            content = []
            file = open(os.path.join(xml_dir, xml_file))
            tree = ET.parse(file)
            root = tree.getroot()

            # 解析file_path
            if len(root.findall("path")[0].text.split("/")) > 2:
                file_name = root.findall("path")[0].text.split("/")[-1]
            else:
                file_name = root.findall("path")[0].text.split("\\")[-1]
            path = image_saved_dir + file_name
            content.append(path)

            # 解析坐标
            for obj in root.iter('object'):
                cls = obj.find('name').text
                classID = classes_dict[cls]
                xmin = obj.find("bndbox").find("xmin").text
                ymin = obj.find("bndbox").find("ymin").text
                xmax = obj.find("bndbox").find("xmax").text
                ymax = obj.find("bndbox").find("ymax").text
                content.append(classID)
                content.append(xmin)
                content.append(ymin)
                content.append(xmax)
                content.append(ymax)
            if count == 0:
                line = " ".join(content)
            else:
                line = "\n" + " ".join(content)
            f.write(line)
            count += 1


if __name__ == "__main__":
    convert_annotation("../data/train_data/xml_file", "train_data.txt")