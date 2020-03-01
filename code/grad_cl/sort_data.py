mags_file = 'mags_resnet18_imagenet.csv'

def read_mags_file(mags_file):
    lines = open(mags_file, "r").readlines()[1:100]
    tup_list = []
    for line in lines:
        parts = line.replace('\n', '').split(',')
        image_name = parts[0]
        mag_grad = float(parts[2])
        tup = (image_name, mag_grad)
        tup_list.append(tup)
    tup_list = sorted(tup_list, key=lambda x: x[1])
    print(tup_list[:10], tup_list[-10:])
    
if __name__ == "__main__":
    read_mags_file(mags_file)
    