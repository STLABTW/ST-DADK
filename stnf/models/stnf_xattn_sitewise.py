"""
STNF-XAttn with Site-wise Temporal Encoding

핵심 아이디어:
1. 각 observed site별로 독립적인 GRU 적용
2. Target은 주변 observed sites의 temporal states를 cross-attention으로 선택
3. Target별로 독립적인 H-step rollout

이 방법이 효과적인 이유:
- 각 사이트가 자신만의 temporal dynamics 학습
- 공간적으로 가까운 사이트는 비슷한 pattern (implicit spatial structure)
- Target은 관련된 observed sites의 정보를 선택적으로 활용
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Literal
import math

from .basis_embedding import BasisEmbedding


class SiteEncoder(nn.Module):
    """각 사이트를 인코딩 (시점별 독립 처리)"""
    
    def __init__(self, p: int, m: int, d: int = 128, dropout: float = 0.0, layernorm: bool = False):
        super().__init__()
        layers = [nn.Linear(p + m + 1, d)]
        if layernorm:
            layers.append(nn.LayerNorm(d))
        layers += [nn.ReLU()]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(d, d), nn.ReLU()]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor, phi: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([X, phi, y_t], dim=-1))


class SiteWiseGRU(nn.Module):
    """각 사이트별로 독립적인 GRU 적용"""
    
    def __init__(self, d: int, h: int = 128, dropout: float = 0.0):
        super().__init__()
        self.d = d
        self.h = h
        
        # 모든 사이트가 공유하는 GRU (parameter efficient)
        self.gru = nn.GRU(d, h, batch_first=True)
        self.do = nn.Dropout(dropout) if (dropout and dropout > 0) else None

    def forward(self, E_seq: torch.Tensor):
        """
        각 observed site별로 GRU 적용
        
        E_seq: (B, L, n_obs, d) - observed site embeddings over time
        
        Returns: 
            H_obs: (B, n_obs, h) - 각 사이트의 temporal state
        """
        B, L, n_obs, d = E_seq.shape
        
        # Reshape: (B, L, n_obs, d) → (B*n_obs, L, d)
        E_flat = E_seq.permute(0, 2, 1, 3).contiguous().view(B * n_obs, L, d)
        
        # GRU: (B*n_obs, L, d) → h_final: (1, B*n_obs, h)
        _, h_final = self.gru(E_flat)
        
        # Reshape back: (1, B*n_obs, h) → (B, n_obs, h)
        H_obs = h_final.squeeze(0).view(B, n_obs, self.h)
        
        if self.do is not None:
            H_obs = self.do(H_obs)
        
        return H_obs


class TargetRollout(nn.Module):
    """Target별로 독립적인 H-step rollout (Autoregressive with h→h projection)"""
    
    def __init__(self, d: int, h: int = 128, m: int = 250, dropout: float = 0.0):
        super().__init__()
        self.h = h
        self.m = m
        
        # Target 위치 정보를 사용해 initial state 생성
        self.spatial_attn = nn.MultiheadAttention(h, num_heads=4, batch_first=True)
        
        # Basis embedding을 h 차원으로 projection
        self.phi_proj = nn.Linear(m, h)
        
        # Rollout을 위한 GRUCell (input dimension = h)
        self.cell = nn.GRUCell(h, h)
        
        # Learnable h → h transformation for autoregressive context
        self.h_to_h = nn.Linear(h, h)
        
        self.do = nn.Dropout(dropout) if (dropout and dropout > 0) else None

    def forward(self, H_obs: torch.Tensor, phi_target: torch.Tensor, H: int):
        """
        H_obs: (B, n_obs, h) - observed sites의 temporal states
        phi_target: (B, n_tar, m) or (n_tar, m) - target basis embeddings
        H: forecast horizon
        
        Returns:
            A: (B, H, n_tar, h) - target별 temporal rollout
        """
        B, n_obs, h = H_obs.shape
        
        # Handle phi_target dimensions
        if phi_target.dim() == 2:
            n_tar = phi_target.shape[0]
            phi_t = phi_target.unsqueeze(0).expand(B, -1, -1)  # (B, n_tar, m)
        else:
            n_tar = phi_target.shape[1]
            phi_t = phi_target
        
        # 1. Initial state: target 위치 기반으로 observed sites에서 선택
        query = self.phi_proj(phi_t)  # (B, n_tar, h)
        h_0, attn_weights = self.spatial_attn(query, H_obs, H_obs)  # (B, n_tar, h)
        
        # 2. Autoregressive Rollout for H steps
        A_list = [h_0]
        h_current = h_0  # (B, n_tar, h)
        
        for step in range(1, H):
            # Transform h → h for autoregressive context (learnable)
            # Reshape for processing: (B, n_tar, h) → (B*n_tar, h)
            h_flat = h_current.contiguous().view(B * n_tar, self.h)
            
            # Generate context from current hidden state
            context = self.h_to_h(h_flat)  # (B*n_tar, h) → (B*n_tar, h)
            
            # GRU step: evolve hidden state
            h_next = self.cell(context, h_flat)  # (B*n_tar, h)
            
            if self.do is not None:
                h_next = self.do(h_next)
            
            # Reshape back: (B*n_tar, h) → (B, n_tar, h)
            h_current = h_next.view(B, n_tar, self.h)
            A_list.append(h_current)
        
        # Stack: list of (B, n_tar, h) → (B, H, n_tar, h)
        A = torch.stack(A_list, dim=1)
        
        return A, attn_weights


class CrossAttnHead(nn.Module):
    """Cross-Attention 기반 예측"""
    
    def __init__(self, h: int, m: int, p: int, d: int = 128, dropout: float = 0.0, 
                 layernorm: bool = False, n_heads: int = 3):
        super().__init__()
        self.n_heads = max(1, int(n_heads))
        self.d_head = int(math.ceil(d / self.n_heads))
        self.d_total = self.d_head * self.n_heads
        
        # Query: [A(h), phi_target(m), X_target(p)] → d_total
        self.proj_q = nn.Linear(h + m + p, self.d_total)
        # Key/Value: H_emb_obs(h) → d_total
        self.proj_k = nn.Linear(h, self.d_total)
        self.proj_v = nn.Linear(h, self.d_total)
        
        # Output MLP
        ff = [nn.Linear(self.d_total, self.d_total)]
        if layernorm:
            ff.append(nn.LayerNorm(self.d_total))
        ff += [nn.ReLU()]
        if dropout and dropout > 0:
            ff.append(nn.Dropout(dropout))
        ff += [nn.Linear(self.d_total, 1)]
        self.mlp = nn.Sequential(*ff)

    def forward(self, A: torch.Tensor, phi_target: torch.Tensor, X_target: torch.Tensor, 
                H_emb_obs: torch.Tensor) -> torch.Tensor:
        """
        A: (B, H, n_tar, h) - target별 temporal rollouts
        phi_target: (n_tar, m) or (B, n_tar, m)
        X_target: (n_tar, p) or (B, n_tar, p) or (H, n_tar, p)
        H_emb_obs: (B, n_obs, h) - observed site temporal states
        
        Returns: y_hat: (B, H, n_tar, 1)
        """
        B, H, n_tar, h = A.shape
        
        # Handle phi_target dimensions
        if phi_target.dim() == 2:
            m = phi_target.shape[1]
            phi = phi_target.unsqueeze(0).unsqueeze(0).expand(B, H, n_tar, m)
        else:
            m = phi_target.shape[2]
            phi = phi_target.unsqueeze(1).expand(B, H, n_tar, m)
        
        # Handle X_target dimensions
        if X_target is None:
            p = 0
            X = torch.zeros(B, H, n_tar, 0, device=A.device)
        elif X_target.dim() == 2:
            p = X_target.shape[1]
            X = X_target.unsqueeze(0).unsqueeze(0).expand(B, H, n_tar, p)
        elif X_target.dim() == 3 and X_target.shape[0] == H:
            p = X_target.shape[2]
            X = X_target.unsqueeze(0).expand(B, -1, -1, -1)
        else:
            p = X_target.shape[2]
            X = X_target.unsqueeze(1).expand(B, H, n_tar, p)
        
        # Query: concat [A, phi, X]
        if p > 0:
            q = torch.cat([A, phi, X], dim=-1)  # (B, H, n_tar, h+m+p)
        else:
            q = torch.cat([A, phi], dim=-1)  # (B, H, n_tar, h+m)
        
        q = self.proj_q(q)  # (B, H, n_tar, d_total)

        # Keys/values from observed-site embeddings
        k = self.proj_k(H_emb_obs)  # (B, n_obs, d_total)
        v = self.proj_v(H_emb_obs)  # (B, n_obs, d_total)
        
        k = k.unsqueeze(1).expand(B, H, -1, -1)
        v = v.unsqueeze(1).expand(B, H, -1, -1)

        # Multi-head attention
        def split_heads(t, n_sites):
            return t.view(B, H, n_sites, self.n_heads, self.d_head)

        qh = split_heads(q, n_tar)
        kh = split_heads(k, k.shape[2])
        vh = split_heads(v, v.shape[2])

        Bh = B * H * self.n_heads
        q2 = qh.permute(0, 1, 3, 2, 4).contiguous().view(Bh, n_tar, self.d_head)
        k2 = kh.permute(0, 1, 3, 2, 4).contiguous().view(Bh, k.shape[2], self.d_head)
        v2 = vh.permute(0, 1, 3, 2, 4).contiguous().view(Bh, v.shape[2], self.d_head)
        
        scores = torch.matmul(q2, k2.transpose(-2, -1)) / (self.d_head ** 0.5)
        att = torch.softmax(scores, dim=-1)
        ctx2 = torch.matmul(att, v2)
        
        ctxh = ctx2.view(B, H, self.n_heads, n_tar, self.d_head).permute(0, 1, 3, 2, 4)
        ctx = ctxh.contiguous().view(B, H, n_tar, self.d_total)
        
        y = self.mlp(ctx)  # (B, H, n_tar, 1)
        return y


class STNFXAttnSiteWise(nn.Module):
    """STNF-XAttn with Site-wise Temporal Encoding
    
    핵심 아이디어:
    1. 각 observed site가 자신만의 temporal dynamics 학습
    2. Target은 주변 observed sites의 정보를 spatial attention으로 선택
    3. Target별로 독립적인 H-step rollout
    """
    
    def __init__(self, p: int = 0, m: int = 64, d_site: int = 128,
                 h_temporal: int = 128, dropout: float = 0.0, layernorm: bool = False,
                 n_heads: int = 3, use_basis_embedding: bool = True,
                 basis_k: int = 250, basis_initialize: Literal['regular'] = 'regular',
                 basis_learnable: bool = False):
        super().__init__()
        self.p = p
        self.use_basis_embedding = use_basis_embedding
        
        if use_basis_embedding:
            self.basis_embedding = BasisEmbedding(
                k=basis_k,
                basis_initialize=basis_initialize,
                basis_learnable=basis_learnable
            )
            self.m = basis_k
        else:
            self.basis_embedding = None
            self.m = 2
        
        # Encode each site at each timestep
        self.encoder = SiteEncoder(self.p, self.m, d_site, dropout=dropout, layernorm=layernorm)
        
        # Site-wise GRU: 각 사이트별 temporal encoding
        self.site_gru = SiteWiseGRU(d_site, h=h_temporal, dropout=dropout)
        
        # Target rollout: target별 H-step prediction
        self.target_rollout = TargetRollout(d_site, h=h_temporal, m=self.m, dropout=dropout)
        
        # Final prediction head
        self.head = CrossAttnHead(h_temporal, self.m, self.p, d=d_site, dropout=dropout,
                                  layernorm=layernorm, n_heads=n_heads)

    def forward(self, obs_coords: torch.Tensor, target_coords: torch.Tensor,
                y_hist_obs: torch.Tensor, H: int,
                X_hist_obs: Optional[torch.Tensor] = None,
                X_fut_target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with site-wise temporal encoding
        """
        B, L, n_obs, _ = y_hist_obs.shape
        
        # Apply basis embedding
        if self.basis_embedding is not None:
            if obs_coords.dim() == 2:
                phi_obs = self.basis_embedding(obs_coords.unsqueeze(0)).squeeze(0)
            else:
                phi_obs = self.basis_embedding(obs_coords)
            
            if target_coords.dim() == 2:
                phi_target = self.basis_embedding(target_coords.unsqueeze(0)).squeeze(0)
            else:
                phi_target = self.basis_embedding(target_coords)
        else:
            phi_obs = obs_coords
            phi_target = target_coords
        
        if phi_obs.dim() == 2:
            phi_obs = phi_obs.unsqueeze(0).expand(B, -1, -1)
        
        if X_hist_obs is None:
            X_hist_obs = torch.zeros(B, L, n_obs, self.p, device=y_hist_obs.device)
        
        # 1. Encode each time slice
        E_list = []
        for t in range(L):
            E_t = self.encoder(X_hist_obs[:, t], phi_obs, y_hist_obs[:, t])
            E_list.append(E_t)
        
        E_seq = torch.stack(E_list, dim=1)  # (B, L, n_obs, d)
        
        # 2. Site-wise temporal encoding: 각 사이트별 GRU
        H_obs = self.site_gru(E_seq)  # (B, n_obs, h)
        
        # 3. Target별 rollout
        A, attn_weights = self.target_rollout(H_obs, phi_target, H)  # (B, H, n_tar, h)
        
        # 4. Prepare X_target
        if phi_target.dim() == 2:
            n_tar = phi_target.shape[0]
        else:
            n_tar = phi_target.shape[1]
        
        if X_fut_target is None:
            X_target = torch.zeros(n_tar, self.p, device=y_hist_obs.device)
        else:
            X_target = X_fut_target
        
        # 5. Final prediction
        y_hat = self.head(A, phi_target, X_target, H_obs)
        
        return y_hat


def create_model(config: dict) -> STNFXAttnSiteWise:
    """설정에서 모델 생성"""
    model = STNFXAttnSiteWise(
        p=config.get('p_covariates', 0),
        d_site=config.get('d_site', 128),
        h_temporal=config.get('h_temporal', 128),
        dropout=config.get('dropout', 0.0),
        layernorm=config.get('layernorm', False),
        n_heads=config.get('n_heads', 4),
        use_basis_embedding=config.get('use_basis_embedding', True),
        basis_k=config.get('basis_k', 250),
        basis_initialize=config.get('basis_initialize', 'regular'),
        basis_learnable=config.get('basis_learnable', False)
    )
    return model
