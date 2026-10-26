from pathlib import Path
import ast
import numpy as np
import pandas as pd

from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

train_images_dir = Path("../data/train/train_images")
test_images_dir = Path("../data/test/test_images")
MAGFILO = Path("../data/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")


def get_image(image_path : str):
    
    """
    _summary_

    Returns:
        _type_: _description_
        
    """

    image_path = train_images_dir / image_path
    image = Image.open(image_path)
    image_array = np.asarray(image.convert("L"))

    return image , image_array


def show_image_with_segmentation(df : pd.DataFrame , image_path: str):

    image , image_array = get_image(image_path)
    plt.figure(figsize=(4,4))
    plt.imshow(image , cmap='gray')
    plt.axis()

    
    segmentation_data = df[df['file_name'] == image_path]['segmentation'].values

    if len(segmentation_data) == 0:
        print(f"No segmentation data found for image {image_path}.")
        return

    
    for seg in segmentation_data:
        if isinstance(seg, str):

            seg = np.array(ast.literal_eval(seg))  
        for polygon in seg:
            polygon = np.array(polygon).reshape(-1, 2)
            plt.gca().add_patch(Polygon(polygon, fill=True, color='red'))
            
    
    plt.show()
    