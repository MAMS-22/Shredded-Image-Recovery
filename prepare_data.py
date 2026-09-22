from environment_and_utilities import *

from sklearn.model_selection import train_test_split
try:
    from keras.utils import to_categorical                      # type: ignore
except ImportError:
    from tensorflow.keras.utils import to_categorical           # type: ignore


def split_data_to_train_test(IM_DIR, data, train_path, test_path, test_size=0.2):
    data = [file for file in data if file[-3:] == 'jpg']
    print('total number of examples:', len(data))
    shuffle(data)
    test_files = data[:int(len(data) * test_size)]
    train_files = data[int(len(data) * test_size):]
    print('test set size:', len(test_files), ' train set size:', len(train_files))
    for file in train_files:
        shutil.copy2(IM_DIR+file, train_path)
    for file in  test_files:
        shutil.copy2(IM_DIR+file, test_path)


def create_legit_combinations(path, tiles_per_dim, limit=10000):
    files = os.listdir(path)
    X = []
    # create images of legit combinations
    for f in files:
        if len(X) >= limit:
            break
        im = cv2.imread(path+f)
        height = im.shape[0]
        width = im.shape[1]
        frac_h = height//tiles_per_dim
        frac_w = width//tiles_per_dim
        for h in range(tiles_per_dim-1):
            for w in range(tiles_per_dim-1):
                crop1 = im[h*frac_h:(h+1)*frac_h, w*frac_w:(w+1)*frac_w, :]

                crop2 = im[(h + 1) * frac_h:(h + 2) * frac_h, w * frac_w:(w + 1) * frac_w, :]
                im1 = np.concatenate((crop1, crop2), axis=0)  # crop1 over crop2
                im1 = np.rot90(im1)
                crop2 = im[ h * frac_h:(h + 1) * frac_h, (w + 1) * frac_w:(w + 2) * frac_w, :]
                im2 = np.concatenate((crop1, crop2), axis=1)  # crop1 to the left of crop2
                # write the images to the "legit" directory
                X += [resize_and_make_border(im1)]
                X += [resize_and_make_border(im2)]
            # the last crop of every line
            crop1 = im[h * frac_h:(h + 1) * frac_h, (tiles_per_dim-1) * frac_w: tiles_per_dim * frac_w, :]
            crop2 = im[(h + 1) * frac_h:(h + 2) * frac_h, (tiles_per_dim-1) * frac_w: tiles_per_dim * frac_w, :]
            im1 = np.concatenate((crop1, crop2), axis=0)  # crop1 over crop2
            im1 = np.rot90(im1)
            X += [resize_and_make_border(im1)]
    return np.array(X)


def create_synthetic_combinations(path, tiles_per_dim, limit=40000):
    files = os.listdir(path)
    X = []
    # create images of NON-legit combinations
    for f in files:
        if len(X) >= limit:
            break
        im = cv2.imread(path + f)
        height = im.shape[0]
        width = im.shape[1]
        frac_h = height // tiles_per_dim
        frac_w = width // tiles_per_dim
        for j in range(tiles_per_dim - 1):
            crop1 = im[j * frac_h:(j + 1) * frac_h, j * frac_w:(j + 1) * frac_w, :]
            for k in range(j + 1, tiles_per_dim):
                crop2 = im[k * frac_h:(k + 1) * frac_h, k * frac_w:(k + 1) * frac_w, :]
                if crop1.shape != crop2.shape:  # make sure both crops are of same size
                    break
                im1 = np.concatenate((crop1, crop2), axis=1)  # crop1 to the left of crop2
                X += [resize_and_make_border(im1)]

                im2 = np.concatenate((crop1, crop2), axis=0)  # crop1 over crop2
                im2 = np.rot90(im2)
                X += [resize_and_make_border(im2)]

                im3 = np.concatenate((crop2, crop1), axis=1)  # crop2 to the left of crop1
                X += [resize_and_make_border(im3)]

                im4 = np.concatenate((crop2, crop1), axis=0)  # crop2 over crop1
                im4 = np.rot90(im4)
                X += [resize_and_make_border(im4)]
    return np.array(X)


def prepare(IM_DIR, tiles_per_dim = 4):
    #files = os.listdir(IM_DIR)
    files = get_n_files(IM_DIR, 1000, 0)
    #print(files)

    os.makedirs(dataset_path)
    os.makedirs(train_path)
    os.makedirs(test_path)

    split_data_to_train_test(IM_DIR, files, train_path, test_path, 0.2)
    X1 = create_legit_combinations(train_path, tiles_per_dim)
    y1 = np.ones((X1.shape[0]))
    X2 = create_synthetic_combinations(train_path, tiles_per_dim)
    y2 = np.zeros((X2.shape[0]))
    print('X1 shape ',X1.shape)
    print('y1 shape ',y1.shape)
    print('X2 shape ',X2.shape)
    print('y2 shape ',y2.shape)

    X = np.concatenate((X1,X2))
    y = np.concatenate((y1,y2))
    y = to_categorical(y, num_classes=2)
    print('X shape ',X.shape)
    print('y shape ',y.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

    np.save(dataset_path+'X_train', X_train)
    np.save(dataset_path+'X_val', X_val)
    np.save(dataset_path+'y_train', y_train)
    np.save(dataset_path+'y_val', y_val)


#prepare(COCO_dataset_path, tiles_per_dim = 4)