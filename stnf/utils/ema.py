"""
Exponential Moving Average (EMA) for PyTorch models
"""
import torch
import torch.nn as nn
from copy import deepcopy


class ModelEMA:
    """
    Exponential Moving Average of model parameters
    
    Maintains a shadow copy of model parameters that is updated with EMA:
    shadow_param = decay * shadow_param + (1 - decay) * param
    
    Usage:
        model = MyModel()
        ema = ModelEMA(model, decay=0.999)
        
        # Training loop
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            ema.update(model)  # Update EMA after optimizer step
        
        # Validation with EMA model
        ema.apply_shadow()  # Temporarily use EMA parameters
        val_loss = model(val_batch)
        ema.restore()  # Restore original parameters
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: PyTorch model to track
            decay: EMA decay rate (higher = slower update, more stable)
                   Typical values: 0.999, 0.9999
        """
        self.decay = decay
        self.model = model
        
        # Create shadow parameters (deep copy of model state)
        self.shadow = {}
        self.backup = {}  # For temporary storage during validation
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """
        Update EMA parameters
        
        Should be called after optimizer.step()
        
        Args:
            model: Current model with updated parameters
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow, f"Parameter {name} not in shadow"
                    new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                    self.shadow[name] = new_average
    
    def apply_shadow(self):
        """
        Temporarily replace model parameters with EMA shadow parameters
        
        Use before validation/inference
        Must call restore() afterwards to restore original parameters
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """
        Restore original model parameters
        
        Use after validation/inference to continue training
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """
        Get EMA state dict (for saving)
        """
        return {
            'decay': self.decay,
            'shadow': self.shadow
        }
    
    def load_state_dict(self, state_dict):
        """
        Load EMA state dict (for resuming)
        """
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
