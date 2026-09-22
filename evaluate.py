from environment_and_utilities import *
from evaluate_aux import solve_image



def evaluate(file_dir, save_path, display=False):
    files = sorted([f for f in os.listdir(file_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    images = []
    for f in files:
        im = cv2.imread(os.path.join(file_dir, f))
        if im is not None:
            images.append(im)

    Y = solve_image(images, display=display, save_path=save_path)
    return Y



def test_image(file_dir, display=False):
    files = sorted([f for f in os.listdir(file_dir) if re.search(r'_(\d+)\.jpg', f)])
    images = []
    for f in files:
        im = cv2.imread(os.path.join(file_dir, f))
        if im is not None:
            images.append(im)

    actual = extract_pos_from_filenames(files)

    Y = solve_image(images, display=display)
    accuracy = 100*(sum([Y[i]==actual[i] for i in range(len(images))])/len(images))
    # print('0-1 Accuracy =', accuracy, '%')
    return accuracy


# test_image()