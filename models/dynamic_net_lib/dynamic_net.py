# models/dynamic_net_lib/utils/dynamic_net.py

import torch
from models.dynamic_net_lib.utils.distributed import CustomDistributedDataParallel

class DynamicNet(object):
    def __init__(self, model, args):
        """
        Initialize the DynamicNet model wrapper.
        
        Args:
            model: The backbone model to wrap
            args: Configuration arguments
        """
        self.model = model
        self.nBlocks = args.nBlocks
        
        # Handle ensemble reweighting properly
        if hasattr(args, 'ensemble_reweight') and args.ensemble_reweight:
            self.reweight = list(map(float, args.ensemble_reweight.split(',')))
        else:
            # Default to 0.5 weighting for each block if not specified
            self.reweight = [0.5] * self.nBlocks
    
    def _get_model(self):
        """
        Helper method to access the correct model instance regardless of wrapping.
        Returns the unwrapped model if it's inside DataParallel, otherwise returns the model directly.
        """
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel, CustomDistributedDataParallel)):
            return self.model.module
        return self.model
        
    def eval(self, stage):
        """
        Set a specific stage of the model to evaluation mode.
        
        Args:
            stage: Index of the stage/block to set to evaluation mode
        """
        model = self._get_model()
        model.blocks[stage].eval()
        model.classifier[stage].eval()
        
    def train(self, stage):
        """
        Set a specific stage of the model to training mode.
        
        Args:
            stage: Index of the stage/block to set to training mode
        """
        model = self._get_model()
        model.blocks[stage].train()
        model.classifier[stage].train()
        
    def cuda_all(self):
        """Move the entire model to CUDA."""
        self.model.cuda()
        return self
        
    def cpu_all(self):
        """Move the entire model to CPU."""
        self.model.cpu()
        return self
        
    def eval_all(self):
        """Set the entire model to evaluation mode."""
        self.model.eval()
        return self
        
    def train_all(self):
        """Set the entire model to training mode."""
        self.model.train()
        return self
        
    def parameters_m(self, stage, separate=False):
        """
        Get parameters for a specific stage of the model.
        
        Args:
            stage: Index of the stage/block
            separate: If True, return block and classifier parameters separately
            
        Returns:
            List of parameters or tuple of parameter lists if separate=True
        """
        model = self._get_model()
        
        if separate:
            return list(model.blocks[stage].parameters()), list(model.classifier[stage].parameters())
        else:
            return list(model.blocks[stage].parameters()) + list(model.classifier[stage].parameters())
            
    def parameters_all(self, stage, all_classifiers=False):
        """
        Get parameters for all stages up to and including the specified stage.
        
        Args:
            stage: Maximum stage index to include
            all_classifiers: If True, include all classifiers up to stage; otherwise just the regressor
            
        Returns:
            List of parameters
        """
        model = self._get_model()
        
        if all_classifiers:
            params = list(model.blocks[:stage + 1].parameters()) + list(model.classifier[:stage + 1].parameters())
        else:
            params = list(model.blocks[:stage + 1].parameters()) + list(model.regressor[stage].parameters())
        return params
        
    def state_dict(self):
        """Get the state dictionary of the model."""
        return {'model': self.model.state_dict()}
        
    def load_state_dict(self, ckpt):
        """
        Load a state dictionary into the model.
        
        Args:
            ckpt: Checkpoint containing the state dictionary
        """
        ckpt = ckpt['model']
        if not hasattr(self.model, 'module'):
            # Handle case where model isn't wrapped but checkpoint was saved from wrapped model
            ckpt = {k.split('module.')[-1] if k.startswith('module.') else k: v for k, v in ckpt.items()}
        self.model.load_state_dict(ckpt)
        
    def forward(self, x):
        """
        Forward pass through all stages of the model with ensemble predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            List of ensemble predictions from each stage
        """
        outs = self.model(x, self.nBlocks)
        preds = [0]
        for i in range(len(outs)):
            # Use reweight parameter from the configuration
            pred = outs[i] + preds[-1] * self.reweight[i]
            preds.append(pred)
        preds = preds[1:]  # Remove the initial 0
        return preds
        
    def forward_all(self, x, stage):
        """
        Forward the model until a specific block and get ensemble predictions.
        
        Args:
            x: Input tensor
            stage: Maximum stage to process
            
        Returns:
            Tuple of (raw outputs, ensemble predictions)
        """
        assert 0 <= stage < self.nBlocks
        outs = self.model(x, stage + 1)  # +1 because we want to include the specified stage
        preds = [0]
        for i in range(len(outs)):
            # Use reweight parameter from the configuration
            pred = outs[i] + preds[-1] * self.reweight[i]
            preds.append(pred)
            if i == stage:
                break
        return outs, preds[1:]  # Return without the initial 0