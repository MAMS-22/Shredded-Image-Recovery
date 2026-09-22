from environment_and_utilities import *
from evaluate import test_image

OUTPUT_DIR = cwd + 'test_images/'
output_2x2_path = OUTPUT_DIR + '2x2/'
output_3x3_path = OUTPUT_DIR + '3x3/'
output_4x4_path = OUTPUT_DIR + '4x4/'
output_5x5_path = OUTPUT_DIR + '5x5/'

def make_shreds(IM_DIR, image_file, tiles_per_dim, folder_index, output_path):
    os.makedirs(output_path+str(folder_index))
    im = cv2.imread(IM_DIR + image_file)
    height = im.shape[0]
    width = im.shape[1]
    frac_h = height // tiles_per_dim
    frac_w = width // tiles_per_dim
    i = 0
    for h in range(tiles_per_dim):
        for w in range(tiles_per_dim):
            crop = im[h * frac_h:(h + 1) * frac_h, w * frac_w:(w + 1) * frac_w]
            cv2.imwrite(output_path + str(folder_index) + '/' + image_file[:-4] + "_{}.jpg".format(i), crop)
            i = i + 1


def prepare_images_crops(IM_DIR, num_of_examples_per_t = 10):
    os.makedirs(OUTPUT_DIR)
    os.makedirs(output_2x2_path)
    os.makedirs(output_3x3_path)
    os.makedirs(output_4x4_path)
    os.makedirs(output_5x5_path)

    files = os.listdir(IM_DIR)
    shuffle(files)

    i = 0
    for f in files[:num_of_examples_per_t]:
        i += 1
        make_shreds(IM_DIR, f, 2, i, output_2x2_path)
        make_shreds(IM_DIR, f, 3, i, output_3x3_path)
        make_shreds(IM_DIR, f, 4, i, output_4x4_path)
        make_shreds(IM_DIR, f, 5, i, output_5x5_path)


def test():

    print('\n***Testing 2x2 images***')
    image_folders = os.listdir(output_2x2_path)
    image_folders = image_folders[:5]
    total_acc = 0
    for image_folder in image_folders:
        acc = test_image(output_2x2_path+image_folder+'/')
        total_acc += acc
        print('image#'+image_folder,' Accuracy =', acc, '%')
    print('***', '2x2 images total accuracy =', total_acc / len(image_folders), '***')

    print('\n***Testing 3x3 images***')
    image_folders = os.listdir(output_3x3_path)
    image_folders = image_folders[:5]
    total_acc = 0
    for image_folder in image_folders:
        acc = test_image(output_3x3_path+image_folder+'/')
        total_acc += acc
        print('image#'+image_folder,' Accuracy =', acc, '%')
    print('***', '3x3 images total accuracy =', total_acc / len(image_folders), '***')

    print('\n***Testing 4x4 images***')
    image_folders = os.listdir(output_4x4_path)
    image_folders = image_folders[:5]
    total_acc = 0
    for image_folder in image_folders:
        acc = test_image(output_4x4_path+image_folder+'/')
        total_acc += acc
        print('image#'+image_folder,' Accuracy =', acc, '%')
    print('***', '4x4 images total accuracy =', total_acc / len(image_folders), '***')
    '''
    print('\n***Testing 5x5 images***')
    image_folders = os.listdir(output_5x5_path)
    image_folders = image_folders[:5]
    total_acc = 0
    for image_folder in image_folders:
        acc = test_image(output_5x5_path+image_folder+'/')
        total_acc += acc
        print('image#'+image_folder,' Accuracy =', acc, '%')
    print('***', '5x5 images total accuracy =', total_acc / len(image_folders), '***')
    '''
#prepare_images_crops(cwd + 'image_dataset/test/', 10)
test()
