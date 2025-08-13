import numpy as np
import os
import cv2
import pickle
import copy


class Layer_Dense:
    def __init__(self, n_inputs,n_neurons,
                 weight_regularizer_l1=0,weight_regularizer_l2=0,
                 bias_regularizer_l1=0,bias_regularizer_l2=0):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        self.biases = np.zeros((1,n_neurons))
        self.weight_regularizer_l1=weight_regularizer_l1
        self.weight_regularizer_l2=weight_regularizer_l2
        self.bias_regularizer_l1=bias_regularizer_l1
        self.bias_regularizer_l2=bias_regularizer_l2
    def forward(self,inputs,training):
        self.inputs = inputs # to store the input value
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients of weights and biases
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights <0]= -1
            self.dweights += self.weight_regularizer_l1 * dL1
        
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
            
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases <0]= -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        
        # Gradient for inputs (to pass to previous layer)
        self.dinputs = np.dot(dvalues, self.weights.T)
    def get_parameters(self):
        return self.weights,self.biases
    def set_parameters(self,weights,biases):
        self.weights=weights
        self.biases = biases

class Layer_Input:
    def forward(self,inputs,training):
        self.output = inputs

class Layer_Dropout:
    def __init__(self,rate):
        self.rate = 1 - rate

    def forward(self,inputs,training):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        
        self.binary_mask = np.random.binomial(1,self.rate,size = inputs.shape) / self.rate

        self.output = inputs * self.binary_mask

    def backward(self,dvalues):
        self.dinputs = dvalues * self.binary_mask

class Activation_ReLU:
    def forward(self,inputs,training):
        self.inputs = inputs # to store the input value
        self.output = np.maximum(0,inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0  # Zero gradient where input was negative

    def predictions(self,outputs):
        return outputs

    
class Activation_Softmax:
    def forward(self, inputs,training):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward ( self , dvalues ):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate ( zip (self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape( - 1 , 1 )
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)

    def predictions(self,outputs):
        return np.argmax(outputs, axis = 1)
        
class Optimizer_SGD :
# Initialize optimizer - set settings,
# learning rate of 1. is default for this optimizer
    def __init__ ( self , learning_rate = 1. , decay = 0. , momentum = 0. ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
# Call once before any parameter updates
    def pre_update_params ( self ):
        if self.decay:
            self.current_learning_rate = self.learning_rate * ( 1. / ( 1. + self.decay * self.iterations))
# Update parameters
    def update_params ( self , layer ):
        # If we use momentum
        if self.momentum:
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr (layer,'weight_momentums' ):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            # Build bias updates
            bias_updates = self.momentum * layer.bias_momentums -self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
            # Vanilla SGD updates (as before momentum update)
        else :
            weight_updates = - self.current_learning_rate * layer.dweights
            bias_updates = - self.current_learning_rate * layer.dbiases
        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params ( self ):
        self.iterations += 1

class Optimizer_Adam :
    # Initialize optimizer - set settings
    def __init__ ( self , learning_rate = 0.001 , decay = 0. , epsilon = 1e-7 ,beta_1 = 0.9 , beta_2 = 0.999 ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    # Call once before any parameter updates
    def pre_update_params ( self ):
        if self.decay:
            self.current_learning_rate = self.learning_rate * ( 1. / ( 1. + self.decay * self.iterations))

    # Update parameters
    def update_params ( self , layer ):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr (layer,'weight_cache' ):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + ( 1 - self.beta_1)*layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + ( 1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / ( 1 - self.beta_1 ** (self.iterations + 1 ))
        bias_momentums_corrected = layer.bias_momentums / ( 1 - self.beta_1 ** (self.iterations + 1 ))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache +( 1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + ( 1 - self.beta_2) * layer.dbiases ** 2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / ( 1 - self.beta_2 ** (self.iterations + 1 ))
        bias_cache_corrected = layer.bias_cache / ( 1 - self.beta_2 ** (self.iterations + 1 ))
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += - self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) +self.epsilon)
        layer.biases += - self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) +self.epsilon)

    # Call once after any parameter updates
    def post_update_params ( self ):
        self.iterations += 1

class Loss:
    
    
    def regularization_loss ( self ):
        # 0 by default
        regularization_loss = 0

        for layer in self.trainable_layers:
            # calculate only when factor greater than 0
            
            if layer.weight_regularizer_l1 > 0 :
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
         
            
            if layer.weight_regularizer_l2 > 0 :
                regularization_loss += layer.weight_regularizer_l2* np.sum(layer.weights * layer.weights)
            
            if layer.bias_regularizer_l1 > 0 :
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
    
            if layer.bias_regularizer_l2 > 0 :
                regularization_loss += layer.bias_regularizer_l2* np.sum(layer.biases * layer.biases)
    
        return regularization_loss
    
    def remember_trainable_layers(self,trainable_layers):
        self.trainable_layers = trainable_layers
    
    def calculate(self,output,y,*,include_regularization = False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        self.accumulated_sum += np.sum(sample_losses) # accumulate sum of losses for the batch
        self.accumulated_count += len(sample_losses) 
        if not include_regularization:
            return data_loss
        return data_loss,self.regularization_loss()
        # Calculates accumulated loss
    def calculate_accumulated ( self ,*, include_regularization = False ):
        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count
        # If just data loss - return it
        if not include_regularization:
            return data_loss
        # Return the data and regularization losses
        return data_loss, self.regularization_loss()
        # Reset variables for accumulated loss
    def new_pass ( self ):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Loss_MSE(Loss):
    def forward(self,y_pred,y_true):
        sample_losses = np.mean((y_true - y_pred)**2,axis = -1)
        return sample_losses

    def backward(self,dvalues,y_true):
        samples=len(dvalues)
        output = len(dvalues[0])
        self.dinputs = -2 *(y_true - dvalues) / output
        self.dinputs = self.dinputs/samples
        
class Loss_categoricalcrossentropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)

        if y_true.ndim == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true ,axis =1)
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = dvalues.shape[1]

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]  # Convert to one-hot encoding

        self.dinputs = -y_true / np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs /= samples  # Normalize by batch size

# Combined softmax activation and cross entropy for faster backward pass
class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__ ( self ):
        self.activation = Activation_Softmax()
        self.loss = Loss_categoricalcrossentropy()
        
    def forward ( self , inputs , y_true ):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backwards(self,dvalues,y_true):
        samples=len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true,axis =1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples),y_true] -=1
        self.dinputs = self.dinputs / samples



class Accuracy :
    # Calculates an accuracy
    # given predictions and ground truth values
        def calculate ( self , predictions , y ):
            
            comparisons = self.compare(predictions, y)
            accuracy = np.mean(comparisons)
            self.accumulated_sum += np.sum(comparisons)
            self.accumulated_count += len (comparisons)
            return accuracy

        def calculate_accumulated ( self ):
            # Calculate an accuracy
            accuracy = self.accumulated_sum / self.accumulated_count
            # Return the data and regularization losses
            return accuracy

        def new_pass ( self ):
            self.accumulated_sum = 0
            self.accumulated_count = 0
                

class Accuracy_Categorical(Accuracy): #Accuracy calculation for classification
    def init(self,y):
        pass
    def compare(self,predictions,y):
        if len(y.shape) == 2:
            y = np.argmax(y,axis = 1)
        return predictions == y

## This is considered alayer in a neural network but doesn’t have weights and biases associated with it. 
## The input layer only contains the training data, 
##and we’ll only use it as a “previous” layer to the first layer during the iteration of the layers in a loop.

class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None
    
    def add(self,layer):
        self.layers.append(layer)
        

    def set(self,*,loss,optimizer,accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count):
            # if it is 1st layer the prvious layer object is input layer
            if i == 0:
                self.layers[i].prev=self.input_layer
                self.layers[i].next=self.layers[i+1]

            elif i < layer_count - 1:
                self.layers[i].prev=self.layers[i-1]
                self.layers[i].next=self.layers[i+1]

            # For last layer - the next object is loss
            #also save aside the reference to last object whose output is the models output
            else:
                self.layers[i].prev=self.layers[i-1]
                self.layers[i].next=self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i],'weights'):
                self.trainable_layers.append(self.layers[i])
            if self.loss is not None :
                self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance (self.layers[ - 1 ], Activation_Softmax) and isinstance (self.loss, Loss_categoricalcrossentropy):
        
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()


    def train(self,X,y,*,epochs = 1,batch_size=None,print_every = 1,validation_data = None):
        
        self.accuracy.init(y)
        train_steps = 1

        if batch_size is not None:
            train_steps = len(X)//batch_size
            if train_steps*batch_size <len(X):
                train_steps +=1
        
        for epoch in range(1,epochs+1):
            self.loss.new_pass()
            self.accuracy.new_pass()
            for step in range(train_steps):

                if batch_size is None:
                    batch_x = X
                    batch_y = y
                else:
                    batch_x = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                
                output = self.forward(batch_x,training = True)
                data_loss,regularization_loss = self.loss.calculate(output,batch_y,include_regularization=True)
                loss = data_loss + regularization_loss
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)
                
                self.backward(output,batch_y)
                
                self.optimizer.pre_update_params()
                
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps -1:
                    print(f'step: {step}, '+
                          f'acc: {accuracy: .3f}, '+
                          f'loss: {loss: .3f}, ' +
                          f'data_loss: {data_loss: .3f}, ' +
                          f'reg_loss: {regularization_loss: .3f} ' +
                          f'lr: {self.optimizer.current_learning_rate}')
            epoch_data_loss,epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization = True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            print(f'training, ' +
                  f'acc: {epoch_accuracy : .3f},'+
                  f'data_loss: {epoch_data_loss : .3f},'+
                  f'reg_loss: {epoch_regularization_loss:.3f},'+
                  f'lr: {self.optimizer.current_learning_rate}')
            if validation_data is not None:
                self.evaluate(*validation_data,batch_size= batch_size)

    def evaluate(self,X_val,y_val,*,batch_size=None): #it is use in training here data is test data to validate
            validation_steps =1
            if batch_size is not None:
                validation_steps = len (X_val) // batch_size
                if validation_steps * batch_size < len (X_val):
                    validation_steps += 1

            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range (validation_steps):
                if batch_size is None :
                    batch_X = X_val
                    batch_y = y_val
                else:
                    batch_X = X_val[step * batch_size:(step + 1 ) * batch_size]
                    batch_y = y_val[step * batch_size:(step + 1 ) * batch_size]

                # Perform the forward pass
                output = self.forward(batch_X,training = False)
                # Calculate the loss
                self.loss.calculate(output, batch_y)
                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                self.accuracy.calculate(predictions, batch_y)

            validation_loss = self.loss.calculate_accumulated()
            validation_accuracy = self.accuracy.calculate_accumulated()
            # Print a summary
            print ( f'validation, '  +
                    f'acc: {validation_accuracy :.3f}, ' +
                    f'loss: {validation_loss :.3f} ' )

    def predict ( self , X ,*, batch_size = None ):
        prediction_steps = 1
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size <len(X):
                prediction_steps += 1

        output = []
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]
            batch_output = self.forward(batch_X,training = False)
            output.append(batch_output)
        return np.vstack(output)

    def forward(self,X,training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output,training)

        return layer.output

    def backward(self,output,y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backwards(output,y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
                
            return
        self.loss.backward(output,y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def get_parameters(self):
        parameters = []

        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters

    

    def set_parameters(self,parameters):
        for parameter_set,layer in zip(parameters,self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self,path):
        with open(path,'wb') as f:
            pickle.dump(self.get_parameters(),f)

    def load_parameters(self,path):
        with open(path,'rb') as f:
            self.set_parameters(pickle.load(f))

    def save(self,path):
        model = copy.deepcopy(self)
        model.loss.new_pass()
        model.accuracy.new_pass()
        model.input_layer.__dict__.pop('output',None)
        model.loss.__dict__.pop('dinputs',None)
        for layer in model.layers:
            for property in ['inputs','output','dinputs','dweights','dbiases']:
                layer.__dict__.pop(property,None)
        with open(path,'wb') as f:
            pickle.dump(model,f)

    @staticmethod
    def load(path):
        with open(path,'rb') as f:
            model = pickle.load(f)
        return model
    
    
import pickle
import numpy as np
import cv2

fashion_mnist_labels = {
    0: 'Zero',
    1: 'One',
    2: 'Two',
    3: 'Three',
    4: 'Four',
    5: 'Five',
    6: 'Six',
    7: 'Seven',
    8: 'Eight',
    9: 'Nine'
}

def load_model(weights_path="mnist_model.pkl"):
    """Load the trained model from pickle."""
    with open(weights_path, "rb") as f:
        model = pickle.load(f)
    return model

def predict_digit(image, model):
    """
    Predict digit from a 28x28 grayscale image.
    This function handles inversion & normalization exactly as in your test code.
    """
    image = cv2.resize(image, (28, 28))
    image = 255 - image  # invert
    image = (image.reshape(1, -1).astype(np.float32) - 127.5) / 127.5


    output =model.forward(image, training=False)
    pred = int(np.argmax(output))
    return pred, fashion_mnist_labels[pred]

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

