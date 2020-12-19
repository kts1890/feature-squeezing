def calculate_accuracy_index(Y_pred, Y_label):
    accurate_index=[]
    for i in range(len(Y_pred)):
        if np.argmax(Y_pred[i]) == np.argmax(Y_label[i]):
            accurate_index.append(i)
    return accurate_index

def calculate_inaccuracy_index(Y_pred, Y_label):
    inaccurate_index=[]
    for i in range(len(Y_pred)):
        if np.argmax(Y_pred[i]) != np.argmax(Y_label[i]):
            inaccurate_index.append(i)
    return inaccurate_index


def show_X(x):
    first_image = x
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

def mappingXY(X_test,Y_pred, Y_test, accurate_index):
    mapping=[]
    for i in accurate_index:       
        mapping.append([X_test[i],np.argmax(Y_pred[i]),np.argmax(Y_test[i]),i])
    return mapping

def classify_map(mapping):
    classifymap=[[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(mapping)):
        classifymap[mapping[i][1]].append(mapping[i][3])
    return classifymap

def exp_density(X_test, data):
    nonzero = 0
    density=[]
    for i in range(len(data)):
        for j in X_test[data[i]]:
            if j != 0:
                nonzero+=1
    density.append(nonzero/len(X_test[0]))
    return density

def exp_edge(X_test, data):
    edge = []
    for i in range(len(data)):
        temp = 0
        for j in X_test[data[i]]:
            for k in range(len(j)):
                if k != 0 and j[k-1]*j[k]==0 and (j[k]+j[k-1])!=0:
                    temp += 1
        edge.append(temp)
    return edge

def save_image(idx, X, result_folder, name):
    examples = X[idx]
    X_list=[]
    X_list.append(X)
    rows = [examples]
    rows += map(lambda x:x[idx], X_list)

    img_fpath = os.path.join(result_folder, '%s.png' % (name) )
    show_imgs_in_rows(rows, img_fpath)
    print ('\n===image is saved in ', img_fpath)
