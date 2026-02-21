import cv2
import torch
import numpy as np
from torchvision import transforms

class ImagePreprocessor:
    def __init__(self, target_size=(512, 512)):
        """
        Initialize preprocessor with target size for resizing
        """
        self.target_size = target_size
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]     # ImageNet std
            )
        ])
    
    def preprocess_frame(self, frame):
        """
        Preprocess a single video frame for CSRNet
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            tensor: Preprocessed PyTorch tensor
            original_shape: Original frame dimensions
        """
        original_shape = frame.shape[:2]  # (height, width)
        
        # Resize frame
        resized = cv2.resize(frame, self.target_size)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Apply transformations and add batch dimension
        tensor = self.transform(rgb_frame).unsqueeze(0)
        
        return tensor, original_shape
    
    def resize_density_map(self, density_map, original_shape):
        """
        Resize density map back to original frame size
        
        Args:
            density_map: Generated density map from CSRNet
            original_shape: Original frame dimensions (height, width)
            
        Returns:
            resized_map: Density map resized to original dimensions
        """
        # Convert to numpy and remove batch dimension
        density_np = density_map.squeeze().cpu().detach().numpy()
        
        # Resize to original shape
        resized_map = cv2.resize(density_np, (original_shape[1], original_shape[0]))
        
        return resized_map
    
    def denormalize_frame(self, tensor):
        """
        Convert normalized tensor back to displayable image
        
        Args:
            tensor: Normalized PyTorch tensor
            
        Returns:
            image: BGR image for OpenCV display
        """
        # Remove batch dimension
        img = tensor.squeeze(0).cpu().detach().numpy()
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img = img * std + mean
        
        # Clip values and convert to uint8
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        # Convert RGB to BGR
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img
