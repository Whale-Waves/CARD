function [y_pred, Z] = CARD_Solver_noC(Bs, missInd, nCluster, param)
%IMVC_CAH_SOLVER_NOC IMVC-CAH solver without explicitly forming C (N x N).
%   Alternates:
%     C-step: Closed-form update of P (anchor space)
%     Y-step: Implicit kernel k-means using P and F
%
%   Input:
%       Bs: Cell array of binary/weight matrices (N_v x m_v)
%       missInd: N x V binary indicator (0=missing)
%       nCluster: Number of clusters
%       param: struct with parameters (lambda, etc.)

    % Parameters
    if ~isfield(param, 'lambda'), param.lambda = 1; end
    if ~isfield(param, 'maxIter'), param.maxIter = 20; end
    
    lambda = param.lambda;
    alpha = 1/(1+lambda);
    beta  = lambda/(1+lambda);
    
    [N, nView] = size(missInd);
    
    % Build global hypergraph incidence H_tilde (N x M_total)
    m_list = cellfun(@(B) size(B, 2), Bs);
    M_total = sum(m_list);
    
    Icell = cell(nView, 1);
    Jcell = cell(nView, 1);
    Vcell = cell(nView, 1);
    col0 = 0;
    
    for v = 1:nView
        B_v = Bs{v}; % N_v x m_v
        [n_v, m_v] = size(B_v);
        
        % Row-normalize anchors
        d_v = sum(B_v, 2);
        d_v(d_v == 0) = eps;
        H_local = spdiags(1./d_v, 0, n_v, n_v) * B_v;
        
        % Map valid parts only
        % missInd(i,v)=1 means observed.
        idx_v = find(missInd(:, v) == 1);
        
        if length(idx_v) ~= n_v
             error('Mismatch between missInd and rows of Bs{%d}', v);
        end
        
        [ii, jj, vv] = find(H_local);
        Icell{v} = idx_v(ii);
        Jcell{v} = col0 + jj;
        Vcell{v} = vv;
        
        col0 = col0 + m_v;
    end
    
    I = vertcat(Icell{:});
    J = vertcat(Jcell{:});
    V = vertcat(Vcell{:});
    H_tilde = sparse(I, J, V, N, M_total);
    [~, M] = size(H_tilde);
    
    % Build Q such that S_H = Q*Q'
    D_vec = sum(H_tilde, 2);
    D_vec(D_vec == 0) = eps;
    D_inv_sqrt = spdiags(1./sqrt(D_vec), 0, N, N);
    
    B_vec = sum(H_tilde, 1)';
    B_vec(B_vec == 0) = eps;
    
    H_hat = D_inv_sqrt * H_tilde;
    B_inv_sqrt = spdiags(sqrt(1./B_vec), 0, M, M);
    Q = H_hat * B_inv_sqrt; % N x M (Sparse)
    
    % --- Phase I: One-shot Initialization ---
    % 1. Compute SVD of Q'Q = V * Sigma^2 * V'
    fprintf('Phase I: Computing spectral geometry of anchors...\n');
    G = full(Q' * Q);
    G = (G + G') / 2;
    [V, D_sq] = eig(G, 'vector'); % D_sq contains sigma_i^2
    
    % Sort eigenvalues descending
    [sigma_sq, idx] = sort(D_sq, 'descend');
    V = V(:, idx);
    
    % Truncate to r = k (as per algorithm, use subscript r)
    r = nCluster;
    sigma_sq = sigma_sq(1:r);
    V = V(:, 1:r);
    M_eff = r;
    
    fprintf('Truncating to r=%d dimensions (as per algorithm)\n', r);
    
    Sigma = sqrt(sigma_sq);     % sigma_i
    Sigma_inv = 1 ./ Sigma;     % 1/sigma_i
    
    % 2. Compute Spectral Filter Omega and Factor E
    
    sigma_sq_col = sigma_sq;
    sigma_sq_row = sigma_sq';
    
    numer = alpha * (1 - alpha) * (sigma_sq_col * sigma_sq_row);
    denom = 1 - alpha * (sigma_sq_col * sigma_sq_row);
    
    % Stability check for denominator
    denom(abs(denom) < 1e-10) = 1e-10;
    
    Omega = numer ./ denom;
    
    % Factorize Omega = E * E'
    % Since Omega is symmetric positive definite (for alpha in [0,1)), we use Eig or Cholesky
    [U_omega, Gam_omega] = eig(Omega, 'vector');
    [gam, sidx] = sort(Gam_omega, 'descend');
    U_omega = U_omega(:, sidx);
    
    % Keep positive components
    pos_idx = gam > 1e-10;
    E = U_omega(:, pos_idx) * diag(sqrt(gam(pos_idx)));
    
    % 3. Initialization using standard spectral clustering approach
    fprintf('Initializing clustering (r=%d dimensions)...\n', r);
    V_scaled = bsxfun(@times, V, Sigma_inv');
    U_init = Q * V_scaled; % N x r (now just k dimensions!)
    % Row normalize (standard practice)
    U_init = bsxfun(@rdivide, U_init, sqrt(sum(U_init.^2, 2)) + eps);
    y_pred = litekmeans(U_init, nCluster, 'Replicates', 10);
    
    % --- Phase II: Alternating Optimization ---
    fprintf('Phase II: Alternating Optimization...\n');
    
    for iter = 1:param.maxIter
        y_old = y_pred;
        
        % 1. Construct Augmented Features Z
        Z = CARD_updateZ(Q, V, y_pred, Sigma_inv, E, alpha, nCluster);
        
        % 2. Compute warm start centers from previous y_pred
        % This ensures K-means refines rather than randomizes
        centers_init = zeros(nCluster, size(Z, 2));
        for c = 1:nCluster
            idx_c = (y_pred == c);
            if sum(idx_c) > 0
                centers_init(c, :) = mean(Z(idx_c, :), 1);
            else
                % Empty cluster: pick a random sample
                centers_init(c, :) = Z(randi(size(Z,1)), :);
            end
        end
        
        % 3. Update Y via K-means on Z with warm start
        y_pred = litekmeans(Z, nCluster, 'Start', centers_init, 'MaxIter', 100);
        
        % Convergence check
        diff = sum(y_pred ~= y_old);
        fprintf('Iter %d/%d: Changed Labels: %d\n', iter, param.maxIter, diff);
        
        if diff == 0
            break;
        end
    end
    
    Z_out = Z; % Return the final features if needed
