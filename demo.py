# Import libraries
import torch, pickle, timm, argparse, streamlit as st
from transformations import get_tfs; from PIL import Image, ImageFont
from utils import get_state_dict; from torchvision.datasets import ImageFolder

# Set the streamlit page config
st.set_page_config(layout = "wide")

def run(args):
    
    """
    
    This function gets parsed arguments and runs the script.
    
    Parameters:
    
        args   - parsed arguments, argparser object;
        
    """
    
    # Get class names for later use
    with open("saved_dls/cls_names.pkl", "rb") as f: cls_names = pickle.load(f)
    
    # Get number of classes
    num_classes = len(cls_names)
    
    # Initialize transformations to be applied
    transformations = get_tfs()
    
    # Set a default path to the image
    default_path = "sample_ims/jellyfish.jpeg"
    
    # Load classification model
    m = load_model(args.model_name, num_classes, args.checkpoint_path)
    st.title("Jellyfish Classifier")
    file = st.file_uploader('Please upload your image')

    # Get image and predicted class
    im, result = predict(m = m, path = file, tfs = transformations, cls_names = cls_names) if file else predict(m = m, path = default_path, tfs = transformations, cls_names = cls_names)
    st.write(f"INPUT IMAGE: "); st.image(im); st.write(f"PREDICTED AS -> {result.upper()}")
        
# @st.cache_data
def load_model(model_name, num_classes, checkpoint_path): 
    
    """
    
    This function gets several parameters and loads a classification model.
    
    Parameters:
    
        model_name      - name of a model from timm library, str;
        num_classes     - number of classes in the dataset, int;
        checkpoint_path - path to the trained model, str;
        
    Output:
    
        m              - a model with pretrained weights and in an evaluation mode, torch model object;
    
    """

    # Initialize model to be trained
    m = timm.create_model(model_name, num_classes = num_classes)
    # Load the pretrained weights
    m.load_state_dict(get_state_dict(args.checkpoint_path))
    
    return m.eval()

def predict(m, path, tfs, cls_names):

    """
    
    This function gets several parameters and makes prediction using the pre-defined model.
    
    Parameters:
    
        m               - an AI model, timm model object;
        path            - path to the pretrained weights, str;
        tfs             - transformations to be applied, torchvision transforms object;
        
    Output:
    
        m              - a model with pretrained weights and in an evaluation mode, torch model object;
    
    """

    # Load an image from the given path
    im = Image.open(path).convert("RGB")
    # Get class names
    cls_names = list(cls_names.keys()) if isinstance(cls_names, dict) else cls_names
    
    return im, cls_names[torch.argmax(m(tfs(im).unsqueeze(0)), dim = 1).item()]
        
if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description = "Imagw Classification Demo Arguments")
    
    # Add arguments
    parser.add_argument("-mn", "--model_name", type = str, default = "rexnet_150", help = "Model name for backbone")
    parser.add_argument("-cp", "--checkpoint_path", type = str, default = "path/to/pretrained/weights", help = "Path to the checkpoint")
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the code
    run(args) 
