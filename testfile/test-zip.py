import zipfile
import tldextract

EXA_1M = '../top-1m.csv.zip'





f = zipfile.ZipFile(EXA_1M, 'r')  # 这里的第二个参数用r表示是读取zip文件，w或a是创建一个zip文件


for x in f.read('top-1m.csv').decode().split()[:100]:
    content = tldextract.extract(x.split(',')[1]).domain
    print(content)
    #print(type(content))



# 读取一个zip压缩包里所有文件的名字。
for f_name in f.namelist():  # z.namelist() 会返回压缩包内所有文件名的列表。
    print(f_name)

