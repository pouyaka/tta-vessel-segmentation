from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class SHOT(nn.Module):
    """SHOT: Do We Really Need to Access the Source Data? 
    Source Hypothesis Transfer for Unsupervised Domain Adaptation
    
    Adapts using entropy minimization with information maximization.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, 
                 ent_weight=0.1, div_weight=1.0):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "SHOT requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.ent_weight = ent_weight
        self.div_weight = div_weight
        
        # Save model and optimizer states
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        
        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer,
                                       self.ent_weight, self.div_weight)
        
        return outputs

    def reset(self):
        """Reset to original model and optimizer states."""
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                self.model_state, self.optimizer_state)


def entropy_loss(pred):
    """Compute entropy for binary segmentation."""
    probs = torch.sigmoid(pred)
    eps = 1e-8
    entropy = -(probs * torch.log(probs + eps) + 
                (1 - probs) * torch.log(1 - probs + eps))
    return entropy.mean()


def diversity_loss(pred):
    """Compute diversity loss (information maximization).
    
    Encourages the model to make diverse predictions across the batch.
    This is done by maximizing the entropy of the average prediction.
    """
    probs = torch.sigmoid(pred)
    # Average prediction across batch
    avg_probs = probs.mean(dim=[0, 2, 3], keepdim=True)
    eps = 1e-8
    # Negative entropy of average (we want to maximize this)
    div_loss = (avg_probs * torch.log(avg_probs + eps) + 
                (1 - avg_probs) * torch.log(1 - avg_probs + eps))
    return div_loss.mean()


@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, ent_weight, div_weight):
    """Forward and adapt model using SHOT objective.
    
    Combines entropy minimization (confident predictions) with
    information maximization (diverse predictions).
    """
    # Forward pass
    outputs = model(x)
    
    # Compute losses
    ent_loss = entropy_loss(outputs)
    div_loss_val = diversity_loss(outputs)
    
    # Combined loss: minimize entropy, maximize diversity
    loss = ent_weight * ent_loss + div_weight * div_loss_val
    
    # Update model
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms."""
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with SHOT."""
    model.train()
    model.requires_grad_(False)
    # Configure BN layers
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatibility with SHOT."""
    is_training = model.training
    assert is_training, "SHOT needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "SHOT needs params to update: check which require grad"
    assert not has_all_params, "SHOT should not update all params: check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "SHOT needs normalization for its optimization"