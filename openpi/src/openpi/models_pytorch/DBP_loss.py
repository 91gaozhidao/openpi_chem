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
    generated_trajectories: torch.Tensor, 
    expert_demonstrations: torch.Tensor, 
    negative_samples: torch.Tensor = None, 
    weight_generated: torch.Tensor = None, 
    weight_expert: torch.Tensor = None,
    weight_negative: torch.Tensor = None, 
    temperature_schedule: tuple[float, ...] = (0.02, 0.05, 0.2)
) -> tuple[torch.Tensor, dict]:
    """
    Computes the Drift-Based Policy (DBP) Loss, leveraging thermodynamic-inspired attractive 
    and repulsive force modeling explicitly operating within the action manifold space.
    
    This loss function enforces generated trajectories to align with expert demonstration manifolds
    while repelling from explicitly defined or inherently generated sub-optimal behavior modes, 
    scaled across a multi-temperature schedule to smoothen the optimization landscape.

    Args:
        generated_trajectories (torch.Tensor): Current policy outputs, shape [Batch, Num_Gen, Seq_Len].
        expert_demonstrations (torch.Tensor): Ground-truth target references, shape [Batch, Num_Expert, Seq_Len].
        negative_samples (torch.Tensor, optional): Explicit negative modes to repel. Defaults to None.
        weight_generated (torch.Tensor, optional): Importance weights for generated samples. Defaults to 1.0.
        weight_expert (torch.Tensor, optional): Importance weights for expert samples. Defaults to 1.0.
        weight_negative (torch.Tensor, optional): Importance weights for negative samples. Defaults to 1.0.
        temperature_schedule (tuple[float, ...]): The sequence of computational temperatures (R).
        
    Returns:
        tuple[torch.Tensor, dict]:
            - loss (torch.Tensor): The aggregated scalar DBP optimization loss per batch.
            - metrics (dict): Diagnostics dictionary capturing gradient scales and force magnitudes across temperatures.
    """
    batch_size, num_generated, sequence_length = generated_trajectories.shape
    num_expert = expert_demonstrations.shape[1]

    # Initialize empty negatives if unlabeled / unsupervised negatives are omitted
    if negative_samples is None:
        negative_samples = generated_trajectories.new_zeros(batch_size, 0, sequence_length)
    num_negative = negative_samples.shape[1]

    # Default to uniform importance weighting if specific priors are unprovided
    if weight_generated is None:
        weight_generated = generated_trajectories.new_ones(batch_size, num_generated)
    if weight_expert is None:
        weight_expert = generated_trajectories.new_ones(batch_size, num_expert)
    if weight_negative is None:
        weight_negative = generated_trajectories.new_ones(batch_size, num_negative)

    # Enforce strict float precision alignment across tensors for stability
    generated_trajectories = generated_trajectories.float()
    expert_demonstrations = expert_demonstrations.float()
    negative_samples = negative_samples.float()
    weight_generated = weight_generated.float()
    weight_expert = weight_expert.float()
    weight_negative = weight_negative.float()

    # Anchor the computational graph to prevent self-referential gradient collapse
    anchored_generated = generated_trajectories.detach()
    
    # Concatenate all spatial targets: [Generated (self), Negatives, Positives (Experts)]
    target_manifold = torch.cat([anchored_generated, negative_samples, expert_demonstrations], dim=1)
    target_weights = torch.cat([weight_generated, weight_negative, weight_expert], dim=1)

    # Compute explicit mathematical goal derivations without retaining autodiff graphs
    with torch.no_grad():
        diagnostics = {}
        
        # 1. Spatial Relationship & Manifold Scaling
        pairwise_distances = compute_pairwise_euclidean_distance(anchored_generated, target_manifold)
        weighted_distances = pairwise_distances * target_weights[:, None, :]
        
        # Compute global spatial scaling factor independent of dimensional sparsity
        global_scale = weighted_distances.mean() / target_weights.mean()
        diagnostics["global_scale"] = global_scale

        # Dimensional normalization to standardize hypersphere distances
        dimensional_scale = torch.clamp(global_scale / (sequence_length ** 0.5), min=1e-3)
        anchored_gen_normalized = anchored_generated / dimensional_scale
        targets_normalized = target_manifold / dimensional_scale

        # Normalize pairwise metrics based on globally approximated scale
        distance_normalized = pairwise_distances / torch.clamp(global_scale, min=1e-3)

        # 2. Prevent collapsing self-connections iteratively (Masking Gen-to-Gen self-loops)
        penalty_mask_value = 100.0
        identity_mask = torch.eye(num_generated, device=generated_trajectories.device, dtype=generated_trajectories.dtype)
        spatial_block_mask = F.pad(identity_mask, (0, num_negative + num_expert))
        spatial_block_mask = spatial_block_mask.unsqueeze(0)
        distance_normalized = distance_normalized + spatial_block_mask * penalty_mask_value

        # 3. Simulate Thermodynamic Gradient Forces via Temperature Schedule
        aggregated_forces = torch.zeros_like(anchored_gen_normalized)

        for temperature in temperature_schedule:
            # Formulate unnormalized softmax logits via inverse temperature scaling
            logits = -distance_normalized / temperature

            # Compute symmetrical attention/affinity across trajectory proximities
            affinity_forward = torch.softmax(logits, dim=-1)
            affinity_backward = torch.softmax(logits, dim=-2)
            
            # Formulate mutual structural affinity to restrict purely directional collapse
            mutual_affinity = torch.sqrt(torch.clamp(affinity_forward * affinity_backward, min=1e-6))
            mutual_affinity = mutual_affinity * target_weights[:, None, :]

            # Partition geometric subsets
            split_boundary = num_generated + num_negative
            affinity_negative_cluster = mutual_affinity[:, :, :split_boundary]
            affinity_expert_cluster = mutual_affinity[:, :, split_boundary:]

            # Derive geometric force vectors (repulsion from negatives vs. attraction to experts)
            sum_expert_attraction = affinity_expert_cluster.sum(dim=-1, keepdim=True)
            repulsive_coefficient = -affinity_negative_cluster * sum_expert_attraction
            
            sum_negative_repulsion = affinity_negative_cluster.sum(dim=-1, keepdim=True)
            attractive_coefficient = affinity_expert_cluster * sum_negative_repulsion

            # Aggregate coefficients along manifold topological boundaries
            force_coefficients = torch.cat([repulsive_coefficient, attractive_coefficient], dim=2)
            total_gradient_force = torch.einsum("biy,byx->bix", force_coefficients, targets_normalized)

            # Centralize constraints over accumulated weights
            accumulated_coeffs = force_coefficients.sum(dim=-1)
            total_gradient_force = total_gradient_force - accumulated_coeffs.unsqueeze(-1) * anchored_gen_normalized

            # Evaluate diagnostic force norm for the current thermodynamic state
            force_magnitude = (total_gradient_force ** 2).mean()
            diagnostics[f"force_magnitude_T{temperature:.2f}"] = force_magnitude

            # Standardize step vectors strictly across variable temperature intervals
            force_scale = torch.sqrt(torch.clamp(force_magnitude, min=1e-8))
            aggregated_forces = aggregated_forces + total_gradient_force / force_scale

        # Formulate ideal target trajectory based on classical gradient descent approximation
        theoretical_target = anchored_gen_normalized + aggregated_forces

    # 4. Perform Gradient Projection (MSE against mathematically constructed objective fields)
    trainable_gen_normalized = generated_trajectories / dimensional_scale.detach()
    spatial_differential = trainable_gen_normalized - theoretical_target.detach()
    
    # Calculate resultant policy error
    dbp_loss = (spatial_differential ** 2).mean(dim=(-1, -2))

    # Average metrics strictly for logging purposes
    diagnostics = {k: v.mean() for k, v in diagnostics.items()}

    return dbp_loss, diagnostics
