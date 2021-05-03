import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import seaborn as sns
import numpy as np

def data_load():
    faces = fetch_lfw_people(min_faces_per_person=60)
    num_images, h, w = faces.images.shape
    X = faces.data
    y = faces.target
    target_names = faces.target_names
    num_classes = target_names.shape[0]
    print("Data loaded.")
    print("Number of Images: ", num_images)
    print("Number of Features: ", X.shape[1])
    print("Number of classes: ", num_classes)

    return X,y,target_names, h , w

def get_model():
    num_components = 150
    pca = PCA(n_components=num_components)
    params = {'C': [1,3, 5, 10], 'gamma': [0.001, 0.005,0.1,10, 1000 ], 'kernel': ['linear', 'rbf'], 'tol': [0.001, 0.0005, 0.0001]}
    svc = SVC()
    classifier = GridSearchCV(svc, params)
    model = make_pipeline(pca, classifier)

    return model

def plot_result(images, titles, names_actual, h, w, n_row=5, n_col=5, fig_title="Prediction Result"):
        fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=1.2, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            ax = fig.add_subplot(n_row, n_col, i + 1)
            ax.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            fc = 'black'
            if titles[i]!=names_actual[i] :
                fc = 'red'
            title = "Predicted: "+titles[i]+"\nActual: "+names_actual[i]
            ax.set_title(title, size=12,color=fc)
            plt.xticks(())
            plt.yticks(())
        if fig_title: 
            fig.suptitle(fig_title+'\n', fontsize=20)

        plt.show(block=True)

if __name__ == "__main__":
    X, y, target_names, h, w = data_load()
    model = get_model()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)
    model = model.fit(X_train, y_train)
    pred = model.predict(X_test)
    plot_result(X_test, target_names[pred], target_names[y_test], h, w)
    print(classification_report(y_test, pred, target_names=target_names))
    conf_mat = confusion_matrix(y_test, pred)
    sns.heatmap(conf_mat, fmt='.2%', cmap='Blues')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig('heatmap.png')