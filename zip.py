import os
import zipfile
data_zip_path = r'C:\Users\default.DESKTOP-A0V01KV\Downloads\가축 행동 영상'
folder_list = ['Training', 'Validation']
save_folder_list = {'Training': 'train', "Validation": 'valid'}
list = os.listdir(data_zip_path)
print(list)
for j in folder_list:
    train_zip_dir = os.path.join(data_zip_path, j)
    print(train_zip_dir)
    for i in os.listdir(train_zip_dir):
        train_zip_dir_ = os.path.join(train_zip_dir, i)
        print(train_zip_dir_)
        with zipfile.ZipFile(train_zip_dir_, 'r') as z:
            z.extractall('E:/Kim/ccc/'+save_folder_list[j]+'/'+i)

