from xml.etree.ElementTree import ElementTree


def get_attribute_from_xml(path_to_file, path_to_attribute):
    tree = ElementTree(file=path_to_file)
    root = tree.getroot()
    object_xml = root.findall(path_to_attribute)
    return object_xml


def get_bound_box_object(path_to_file):
    bounded_box_object = get_attribute_from_xml(path_to_file, "object/bndbox/*")
    bounded_box_dict = {}
    for i in bounded_box_object:
        bounded_box_dict[i.tag] = i.text
    return bounded_box_dict