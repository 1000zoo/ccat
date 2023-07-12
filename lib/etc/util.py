import os

def check_dir(path: str):
    path = path.replace("\\", '/')  # 윈도우 경우
    pp = path.split('/')
    temp = '/' + pp.pop(0) if not pp[0] else './'

    while pp:
        tt = pp.pop(0)
        ss = os.path.join(temp, tt)
        if not tt in os.listdir(temp):
            os.mkdir(ss)

        temp = ss

if __name__ == '__main__':
    pass