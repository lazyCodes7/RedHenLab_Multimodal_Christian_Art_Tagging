import torch.nn as nn
import torch
from transformers import ViTForImageClassification
class ArtDLClassifier(nn.Module):
    '''
        About:
            ViT Classifier that acts as a feature extractor.
        
        Inputs:
            c1_types - no of out classes of class_type 1
            c2_types - no of out classes of class_type 2
            c3_types - no of out classes of class_type 3
        
        Methods:
            1. forward(self, x)
                About:
                    Implementation of forward propagation of the network defined.
                    
                Inputs: 
                    x - the image to be used for inference/training
                
                Outputs:
                    result_dict = {
                        "mary" -> linear layer outputs for classification of mary
                        "male_saint" -> linear layer outputs for classification of male saints
                        "female_saint" -> linear layer outputs for classification of female saints
                        "hidden_output" -> outputs the hidden state of the transformers
                    }
            
    '''
    def __init__(self, c1_types, c2_types, c3_types):

        super().__init__()
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', output_hidden_states = True)
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 50)

        self.mary = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=50, out_features=c1_types)
        )
        self.male_saint = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=50, out_features=c2_types)
        )
        self.female_saint = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=50, out_features=c3_types)
        )

    def forward(self, x):
        model_result = self.model(x)
        hidden_output = model_result.hidden_states[-1]

        x = self.fc1(model_result.logits)
        x = self.fc2(x)
        return {
            'mary': self.mary(x),
            'male_saint': self.male_saint(x),
            'female_saint': self.female_saint(x),
            'hidden_output' : hidden_output
        }