from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, Input, concatenate, Flatten, SpatialDropout1D, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from keras import regularizers


class NNForestClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self, output_dim, n_estimators=10, validation_split=0.0, shuffle = True):
        self.n_estimators = n_estimators
        self.validation_split = validation_split 
        self.output_dim = output_dim
        self.shuffle = shuffle
        self.opt = Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        
    def x_preprocessing(self, X):
        leaf_indicies = self.RF_fit.apply(X)
        leaf_indicies = leaf_indicies.transpose()
        return leaf_indicies
    
    def tree_preprocessing(self, tree):
        self.label_enc = LabelEncoder()
        tree_label = self.label_enc.fit(tree)
        int_enc = self.label_enc.transform(tree)
        return int_enc, tree_label
        
    def fit(self, X, y):
        if self.shuffle:
            self.p = np.random.permutation(len(y))
            print(len(y), y)
            X, y = X[self.p], y[self.p]
            print(y)
        #pdb.set_trace()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.validation_split, shuffle=False)
        self.RF_fit = RandomForestClassifier(n_estimators=self.n_estimators)
        self.RF_fit.fit(X_train, y_train)
        leaf_indicies = self.x_preprocessing(X)
        self.y_enc = LabelEncoder()
        self.y_enc.fit(y)
        num_y = self.y_enc.transform(y)
        new_y = to_categorical(num_y, num_classes=len(set(num_y)))
        y_dim = new_y.shape[1]
        
        self.tree_labels = []
        new_X = []
        inputs = []
        embedded_inputs = []
        for i, fit_tree in enumerate(leaf_indicies):
            int_enc, tree_label = self.tree_preprocessing(fit_tree)
            self.tree_labels.append(tree_label)
            new_X.append(int_enc)
            # + 2 for 1 to index and 1 tree routes not in the traning set
            input_dim = np.amax(int_enc) + 2
            #main_input = Input(shape=(1,), dtype='int8')
            main_input = Input(shape=(1,), name="input%d" % (i))
            inputs.append(main_input)
            in_emb = Embedding(input_dim=input_dim, output_dim=self.output_dim, input_length=1)(main_input)
            embedded_inputs.append(in_emb)
        x = concatenate(embedded_inputs)
        #x = SpatialDropout1D(.25)(x)
        #x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(15)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(8)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        if(y_dim == 1):
            main_output = Dense(1, activation='sigmoid', name='main_output')(x)
            loss = 'binary_crossentropy'
        elif(y_dim > 1):
            print("categorical")
            main_output = Dense(y_dim, activation='softmax')(x)
            loss = 'categorical_crossentropy'
            
        self.model = Model(inputs = inputs, outputs = main_output)
        self.model.compile(loss=loss, optimizer=self.opt, metrics=['accuracy'])
        #print(self.model.summary())
        self.history = self.model.fit(new_X, new_y, epochs=70, verbose = 1, batch_size = 50, validation_split=self.validation_split)
        
    def predict(self, X):
        probs = self.predict_prob(X)
        predictions=probs.round().astype(int)
        return self.y_enc.inverse_transform(np.argmax(predictions, axis=1))
    
    def predict_prob(self, X):
        pred_leaf = self.x_preprocessing(X)
        new_pred_X = []
        for i, pred_tree in enumerate(pred_leaf):
            labels = self.tree_labels[i].classes_
            diff = set(pred_tree).difference(set(labels))
            dic_trans = {a: b for b, a in enumerate(labels)}
            for j in diff:
                dic_trans[j] = np.amax(labels) + 1
            pred_enc = [dic_trans.get(n, n) for n in pred_tree]
            new_pred_X.append(np.array(pred_enc))
        return self.model.predict(new_pred_X, verbose=1)
    
    
              

