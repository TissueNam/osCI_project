import torch
import torch.nn as nn
from fuxin_network32 import classifier32
from torchsummary import summary

# Set Options
NUM_CLASSES = 6
CLASSIFIER_EPOCHS = 1
BATCH_SIZE = 64
LATENT_SIZE = 20

def load_training_dataset():
    
    
def main(): 
    # 1. Train a Classifier on the K known classes
    print("1. Train a Classifier on the K known classes")
    classifier = classifier32(num_classes=NUM_CLASSES, latent_size=LATENT_SIZE)
    #summary(classifier, (3, 32, 32)) # visualize network model
    for i in range(CLASSIFIER_EPOCHS):
        load_training_dataset()
        
        
    # 2. Train the counterfactual generative model
    
    # 3. Generative counterfactual open set image
    
    # 4. Use counterfactual open set images to re-train the classifier
    
    # 5. Output ROC curve comparing the methods
    
if __name__ == '__main__':
    main()