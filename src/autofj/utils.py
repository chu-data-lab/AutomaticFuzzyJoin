from datetime import datetime
import os

def print_log(message):
    print("{}:{}".format(datetime.now().strftime('%H:%M:%S'), message))

def makedir(dir_list, file=None, remove_old_dir=False):
    save_dir = os.path.join(*dir_list)

    if remove_old_dir and os.path.exists(save_dir) and file is None:
        shutil.rmtree(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if file is not None:
        save_dir = os.path.join(save_dir, file)
    return save_dir