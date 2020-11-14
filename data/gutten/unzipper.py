import os
from zipfile import ZipFile


walk = os.walk(os.path.curdir)
abspath = os.path.abspath(os.path.curdir)
extract_path = os.path.join(abspath, "morgen")

zips = [os.path.join(os.path.join(abspath, root), xi) for root, *rest in walk for x in rest for xi in x if ".zip" in xi]

for zip in zips:
    with ZipFile(zip, "r") as z:
        z.extractall("morgen")

# def extract(path):
#     if os.path.isdir(path):
#         res = []
#         contents = os.listdir(path)
#         for c in contents:
#             if not "-" in c:
#                 res.append(extract(os.path.join(path, c)))
#         return res
#     else:
#         return path
# 
# res = []
# for x in os.listdir(extract_path):
#     xd = os.path.join(extract_path, x)
#     if os.path.isdir(xd):
#         res.extend(extract(xd))
#     else: res.append(xd)

