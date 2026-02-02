function Z = CARD_updateZ(Q, V, y_pred, Sigma_inv, E, alpha, nCluster)
%IMVC_CAH_UPDATEZ Construct Augmented Features Z(Y) for CARD.
%   Z(Y) = [ Q * T,  sqrt(1-alpha) * Y_tilde ]
%
%   Input:
%       Q: N x M sparse matrix
%       V: M x M_eff dense singular vectors
%       y_pred: N x 1 cluster labels
%       Sigma_inv, E, alpha, nCluster
%
%   Output:
%       Z: N x (Rank_Omega * c + c) dense matrix

    N = size(Q, 1);
    [M_eff, Rank_E] = size(E);
    
    % 1. Construct Normalized Indicator Y_tilde (N x c)
    Y_mat = sparse(1:N, y_pred, 1, N, nCluster);
    n_k = full(sum(Y_mat, 1));
    inv_sqrt_nk = 1 ./ sqrt(n_k + eps);
    Y_tilde = full(Y_mat) .* inv_sqrt_nk; 
    
    % 2. Calculate Projected Centroids J (M_eff x c)
    R = Q' * Y_tilde;
    R_proj = V' * R;
    J = bsxfun(@times, Sigma_inv, R_proj);
    
    % 3. Construct Structural Projection P(Y)
    T_spectral = zeros(M_eff, Rank_E * nCluster);
    for r = 1:Rank_E
        e_r = E(:, r);
        Block = bsxfun(@times, e_r, J);
        col_start = (r-1) * nCluster + 1;
        col_end   = r * nCluster;
        T_spectral(:, col_start:col_end) = Block;
    end
    
    % 4. Construct Z_struct = Q * Phi * P(Y)
    Phi = bsxfun(@times, V, Sigma_inv'); 
    T_full = Phi * T_spectral;
    Z_struct = Q * T_full;
    
    % 5. Construct Z_cluster
    scale = sqrt(1 - alpha);
    Z_cluster = scale * Y_tilde;
    
    % 6. Concatenate
    Z = [Z_struct, Z_cluster];

end
