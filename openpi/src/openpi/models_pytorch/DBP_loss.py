import torch
import torch.nn.functional as F


def compute_pairwise_euclidean_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the pairwise Euclidean (L2) distance between two sets of multidimensional representations.
    
    This function utilizes the computationally efficient dot-product expansion formulation:
        ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * <x, y>
        
    Args:
        x (torch.Tensor): The first set of representations, shape [Batch, N, Dim].
        y (torch.Tensor): The second set of representations, shape [Batch, M, Dim].
        eps (float): A small numerical constant to prevent NaN gradients during backpropagation 
                     through the square root at zero distance. Defaults to 1e-8.
                     
    Returns:
        torch.Tensor: Pairwise distance matrix of shape [Batch, N, M].
    """
    xy_dot_product = torch.einsum("bnd,bmd->bnm", x, y)
    x_squared_norms = torch.einsum("bnd,bnd->bn", x, x)
    y_squared_norms = torch.einsum("bmd,bmd->bm", y, y)
    
    # Utilizing broadcasting to compute the expanded squared distance matrix
    squared_distance = x_squared_norms[:, :, None] + y_squared_norms[:, None, :] - 2 * xy_dot_product
    
    # Clamp to strictly positive values to ensure numerical stability before square root operation
    return torch.sqrt(torch.clamp(squared_distance, min=eps))


def compute_dbp_loss(
    preds: torch.Tensor, 
    pos_targets: torch.Tensor, 
    neg_targets: torch.Tensor = None, 
    w_pred: torch.Tensor = None, 
    w_pos: torch.Tensor = None,
    w_neg: torch.Tensor = None, 
    temp_schedule: tuple[float, ...] = (0.02, 0.05, 0.2)
) -> tuple[torch.Tensor, dict]:
    """
    Computes the Drift-Based Policy (DBP) Loss, leveraging thermodynamic-inspired attractive 
    and repulsive force modeling explicitly operating within the action manifold space.
    
    This loss function enforces generated trajectories to align with expert demonstration manifolds
    while repelling from explicitly defined or inherently generated sub-optimal behavior modes, 
    scaled across a multi-temperature schedule to smoothen the optimization landscape.

    Args:
        preds (torch.Tensor): Current policy outputs, shape [Batch, Num_Gen, Seq_Len].
        pos_targets (torch.Tensor): Ground-truth target references, shape [Batch, Num_Expert, Seq_Len].
        neg_targets (torch.Tensor, optional): Explicit negative modes to repel. Defaults to None.
        w_pred (torch.Tensor, optional): Importance weights for generated samples. Defaults to 1.0.
        w_pos (torch.Tensor, optional): Importance weights for expert samples. Defaults to 1.0.
        w_neg (torch.Tensor, optional): Importance weights for negative samples. Defaults to 1.0.
        temp_schedule (tuple[float, ...]): The sequence of computational temperatures (R).
        
    Returns:
        tuple[torch.Tensor, dict]:
            - loss (torch.Tensor): The aggregated scalar DBP optimization loss per batch.
            - metrics (dict): Diagnostics dictionary capturing gradient scales and force magnitudes across temperatures.
    """
    batch_size, num_preds, seq_len = preds.shape
    num_pos = pos_targets.shape[1]

    # Initialize empty negatives if unlabeled / unsupervised negatives are omitted
    if neg_targets is None:
        neg_targets = preds.new_zeros(batch_size, 0, seq_len)
    num_neg = neg_targets.shape[1]

    # Default to uniform importance weighting if specific priors are unprovided
    if w_pred is None:
        w_pred = preds.new_ones(batch_size, num_preds)
    if w_pos is None:
        w_pos = preds.new_ones(batch_size, num_pos)
    if w_neg is None:
        w_neg = preds.new_ones(batch_size, num_neg)

    # Enforce strict float precision alignment across tensors for stability
    preds = preds.float()
    pos_targets = pos_targets.float()
    neg_targets = neg_targets.float()
    w_pred = w_pred.float()
    w_pos = w_pos.float()
    w_neg = w_neg.float()

    # Anchor the computational graph to prevent self-referential gradient collapse
    anchored_preds = preds.detach()
    
    # Concatenate all spatial targets: [Generated (self), Negatives, Positives (Experts)]
    all_targets = torch.cat([anchored_preds, neg_targets, pos_targets], dim=1)
    all_weights = torch.cat([w_pred, w_neg, w_pos], dim=1)

    # -------------------------------------------------------------------------
    # Phase 1: Spatial Relationship & Manifold Scaling
    # -------------------------------------------------------------------------
    with torch.no_grad():
        diagnostics = {}
        
        # Calculate pairwise Euclidean distances between generated predictions and all target modalities
        # dists shape: [Batch, Num_Preds, Num_Preds + Num_Neg + Num_Pos]
        dists = compute_pairwise_euclidean_distance(anchored_preds, all_targets)
        
        # Apply importance weights (w_pred, w_neg, w_pos) to the calculated distances
        weighted_dists = dists * all_weights[:, None, :]
        
        # Compute global spatial scaling factor (approximates the diameter of the current manifold)
        # We normalize by the mean weight to ensure invariance to the absolute magnitude of weights
        scale = weighted_dists.mean() / all_weights.mean()
        diagnostics["global_scale"] = scale

        # Establish a dimensional scaling factor to correct for the curse of dimensionality
        # sequence_length acts as the dimension count of the trajectory vector
        dim_scale = torch.clamp(scale / (seq_len ** 0.5), min=1e-3)
        
        # Normalize representations to project them into a roughly unit-scale structural space
        anchored_preds_norm = anchored_preds / dim_scale
        targets_norm = all_targets / dim_scale

        # Normalize pairwise distance metrics based on the globally approximated scale
        dists_norm = dists / torch.clamp(scale, min=1e-3)

        # -------------------------------------------------------------------------
        # Phase 2: Topology Regularization (Prevent self-loops)
        # -------------------------------------------------------------------------
        # We must prevent 'Prediction A' from forming an attractive force towards itself.
        # We add a massive penalty (100.0) exclusively to the diagonal of the Prediction-to-Prediction distance matrix.
        penalty_mask_value = 100.0
        identity_mask = torch.eye(num_preds, device=preds.device, dtype=preds.dtype)
        # Pad the mask to cover the shape [Num_Preds, Num_Preds + Num_Neg + Num_Pos]
        # Padding format: (pad_last_dim_left, pad_last_dim_right) -> pad 0 on left, Num_Neg + Num_Pos on right
        spatial_block_mask = F.pad(identity_mask, (0, num_neg + num_pos)).unsqueeze(0)
        dists_norm = dists_norm + spatial_block_mask * penalty_mask_value

        # -------------------------------------------------------------------------
        # Phase 3: Thermodynamic Gradient Force Simulation across Temperatures
        # -------------------------------------------------------------------------
        aggregated_forces = torch.zeros_like(anchored_preds_norm)

        for temperature in temp_schedule:
            # 3.1: Formulate unnormalized softmax logits via inverse temperature scaling (T)
            # Smaller T focuses strictly on nearest neighbors; Larger T sees the global manifold structures
            logits = -dists_norm / temperature

            # 3.2: Compute symmetrical attention/affinity matrices
            # affinity_forward: How strongly a Prediction is attracted to a Target (dim=-1)
            # affinity_backward: How strongly a Target attracts a Prediction (dim=-2)
            affinity_forward = torch.softmax(logits, dim=-1)
            affinity_backward = torch.softmax(logits, dim=-2)
            
            # 3.3: Calculate Mutual Structural Affinity
            # Geometric mean explicitly restricts purely unidirectional collapse (e.g., all preds collapsing to 1 strong expert)
            mutual_affinity = torch.sqrt(torch.clamp(affinity_forward * affinity_backward, min=1e-6))
            mutual_affinity = mutual_affinity * all_weights[:, None, :]

            # 3.4: Partition topological subsets (Negative Modes vs. Expert Positives)
            split_boundary = num_preds + num_neg
            affinity_neg_cluster = mutual_affinity[:, :, :split_boundary]  # Includes self-pred and negative forces
            affinity_pos_cluster = mutual_affinity[:, :, split_boundary:]  # Includes expert target forces

            # 3.5: Formulate thermodynamic push-pull coefficients
            # Experts exert highly attractive forces proportionate to how far away negative boundaries are
            sum_pos_attraction = affinity_pos_cluster.sum(dim=-1, keepdim=True)
            repulsive_coeff = -affinity_neg_cluster * sum_pos_attraction
            
            # Negatives exert repulsive forces proportionate to how strong the expert attraction is
            sum_neg_repulsion = affinity_neg_cluster.sum(dim=-1, keepdim=True)
            attractive_coeff = affinity_pos_cluster * sum_neg_repulsion

            # Concatenate generalized coefficients along the target manifold elements
            force_coeffs = torch.cat([repulsive_coeff, attractive_coeff], dim=2)
            
            # 3.6: Synthesize true structural gradient forces via linear combinations 
            # Multiplies coefficients by standard target vectors to establish a directional pull
            total_gradient_force = torch.einsum("biy,byx->bix", force_coeffs, targets_norm)

            # Centralize constraints over accumulated weights to eliminate residual coordinate shifts
            accumulated_coeffs = force_coeffs.sum(dim=-1)
            total_gradient_force = total_gradient_force - accumulated_coeffs.unsqueeze(-1) * anchored_preds_norm

            # 3.7: Evaluate diagnostic localized force magnitude and accumulate step gradients
            force_magnitude = (total_gradient_force ** 2).mean()
            diagnostics[f"force_magnitude_T{temperature:.2f}"] = force_magnitude

            # Standardize step vectors across varying T intervals so high-temp forces don't overpower local low-temp forces
            force_scale = torch.sqrt(torch.clamp(force_magnitude, min=1e-8))
            aggregated_forces = aggregated_forces + total_gradient_force / force_scale

        # Formulate ideal optimization target based on classical simulated gradient descent logic
        theoretical_target = anchored_preds_norm + aggregated_forces

    # -------------------------------------------------------------------------
    # Phase 4: Construct the End-to-End Pytorch MSE Backpropagation Graph
    # -------------------------------------------------------------------------
    # Re-attach trainable normalized predictions to allow AutoGrad to differentiate through the model
    trainable_preds_norm = preds / dim_scale.detach()
    
    # Calculate difference between predictions and the theoretically formulated targets
    spatial_diff = trainable_preds_norm - theoretical_target.detach()
    
    # Calculate resultant scalar policy error per item in the batch
    # Reduces over Number of Gen (dim=-2) and Sequence Length / Dimensions (dim=-1)
    dbp_loss = (spatial_diff ** 2).mean(dim=(-1, -2))

    # Average metrics strictly for logging purposes
    diagnostics = {k: v.mean() for k, v in diagnostics.items()}

    return dbp_loss, diagnostics
