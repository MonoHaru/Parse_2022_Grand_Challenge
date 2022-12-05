# train dataset uuzip
import os
os.chdir('E:\\Parse2022')
import gzip
import shutil

train_all_folder = os.listdir('E:\\Parse2022\\train')
read_dir = 'E:\\Parse2022/train'
save_dir = 'E:\\Parse2022\\dataset\\train'

for folder_name in train_all_folder:
    for im_la in ['image', 'label']:
        train_zip_dir = os.path.join(read_dir, folder_name, im_la)
        gzip_file = os.listdir(train_zip_dir)
        read_path = os.path.join(train_zip_dir, gzip_file[0])
        save_path = os.path.join(save_dir, folder_name, im_la)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, gzip_file[0])
        with gzip.open(read_path, 'rb') as f_in:
            with open(save_path[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
