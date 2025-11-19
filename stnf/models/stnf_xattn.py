"""
STNF-XAttn: Spatio-Temporal Neural Field with Cross-Attention (v3)
원본 구조로 복원 - 잘 작동했던 버전
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
        """
        X: (B, n, p) - covariates
        phi: (B, n, m) - basis embedding
        y_t: (B, n, 1) - observation at time t
        
        Returns: (B, n, d)
        """
        return self.mlp(torch.cat([X, phi, y_t], dim=-1))


class GRURoll(nn.Module):
    """GRU 인코딩 및 롤아웃 (개선: Autoregressive with h→d projection)"""
    
    def __init__(self, d: int, h: int = 128, dropout: float = 0.0):
        super().__init__()
        self.d = d
        self.h = h
        self.gru = nn.GRU(d, h, batch_first=True)
        self.cell = nn.GRUCell(d, h)  # input dimension = d
        
        # Learnable h → d transformation
        self.h_to_d = nn.Linear(h, d)
        
        self.do = nn.Dropout(dropout) if (dropout and dropout > 0) else None

    def forward(self, E_seq: torch.Tensor, H: int):
        """
        E_seq: (B, L, d) - 시간별 평균 임베딩
        H: 예측 horizon
        
        Returns: (B, H, h)
        """
        _, h = self.gru(E_seq)  # h: (1, B, h)
        h = h[0]  # (B, h)
        
        outs = []
        h_current = h
        for _ in range(H):
            # Transform h → d (learnable projection)
            context = self.h_to_d(h_current)  # (B, h) → (B, d)
            
            # GRU step with evolved context
            h_next = self.cell(context, h_current)
            if self.do is not None:
                h_next = self.do(h_next)
            outs.append(h_next)
            h_current = h_next  # Autoregressive update
        
        return torch.stack(outs, dim=1)  # (B, H, h)


class CrossAttnHead(nn.Module):
    """Cross-Attention 기반 예측"""
    
    def __init__(self, h: int, m: int, p: int, d: int = 128, dropout: float = 0.0, 
                 layernorm: bool = False, n_heads: int = 3):
        super().__init__()
        self.n_heads = max(1, int(n_heads))
        # Per-head dimension
        self.d_head = int(math.ceil(d / self.n_heads))
        self.d_total = self.d_head * self.n_heads
        
        # Query: [A(h), phi_target(m), X_target(p)] → d_total
        self.proj_q = nn.Linear(h + m + p, self.d_total)
        # Key/Value: H_emb_obs(d) → d_total
        self.proj_k = nn.Linear(d, self.d_total)
        self.proj_v = nn.Linear(d, self.d_total)
        
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
        A: (B, H, h) - temporal rollouts
        phi_target: (n_tar, m) or (B, n_tar, m) - target site basis
        X_target: (n_tar, p) or (B, n_tar, p) - target site covariates
        H_emb_obs: (B, n_obs, d) - observed site embeddings
        
        Returns: y_hat: (B, H, n_tar, 1)
        """
        B, H, h = A.shape
        
        # Handle phi_target dimensions
        if phi_target.dim() == 2:
            # (n_tar, m)
            n_tar, m = phi_target.shape
            # Expand to (B, H, n_tar, m)
            phi = phi_target.unsqueeze(0).unsqueeze(0).expand(B, H, n_tar, m)
        else:
            # (B, n_tar, m)
            n_tar, m = phi_target.shape[1], phi_target.shape[2]
            # Expand to (B, H, n_tar, m)
            phi = phi_target.unsqueeze(1).expand(B, H, n_tar, m)
        
        # Handle X_target dimensions
        if X_target is None:
            p = 0
            X = torch.zeros(B, H, n_tar, 0, device=A.device)
        elif X_target.dim() == 2:
            # (n_tar, p)
            p = X_target.shape[1]
            X = X_target.unsqueeze(0).unsqueeze(0).expand(B, H, n_tar, p)
        elif X_target.dim() == 3 and X_target.shape[0] == H:
            # (H, n_tar, p) - time-varying
            p = X_target.shape[2]
            X = X_target.unsqueeze(0).expand(B, -1, -1, -1)
        else:
            # (B, n_tar, p)
            p = X_target.shape[2]
            X = X_target.unsqueeze(1).expand(B, H, n_tar, p)
        
        # Query: concat [A, phi, X]
        # A: (B, H, h) → (B, H, n_tar, h)
        A_exp = A.unsqueeze(2).expand(B, H, n_tar, h)
        
        if p > 0:
            q = torch.cat([A_exp, phi, X], dim=-1)  # (B, H, n_tar, h+m+p)
        else:
            q = torch.cat([A_exp, phi], dim=-1)  # (B, H, n_tar, h+m)
        
        q = self.proj_q(q)  # (B, H, n_tar, d_total)

        # Keys/values from observed-site embeddings
        k = self.proj_k(H_emb_obs)  # (B, n_obs, d_total)
        v = self.proj_v(H_emb_obs)  # (B, n_obs, d_total)
        
        # Expand to horizon dimension
        k = k.unsqueeze(1).expand(B, H, -1, -1)  # (B, H, n_obs, d_total)
        v = v.unsqueeze(1).expand(B, H, -1, -1)  # (B, H, n_obs, d_total)

        # Reshape into heads
        def split_heads(t, n_sites):
            # (B, H, n_sites, d_total) → (B, H, n_sites, n_heads, d_head)
            return t.view(B, H, n_sites, self.n_heads, self.d_head)

        qh = split_heads(q, n_tar)  # (B, H, n_tar, heads, d_head)
        kh = split_heads(k, k.shape[2])  # (B, H, n_obs, heads, d_head)
        vh = split_heads(v, v.shape[2])  # (B, H, n_obs, heads, d_head)

        # Compute attention per head using batched matmul
        Bh = B * H * self.n_heads
        # Reshape: (B, H, n_sites, heads, d_head) → (B*H*heads, n_sites, d_head)
        q2 = qh.permute(0, 1, 3, 2, 4).contiguous().view(Bh, n_tar, self.d_head)
        k2 = kh.permute(0, 1, 3, 2, 4).contiguous().view(Bh, k.shape[2], self.d_head)
        v2 = vh.permute(0, 1, 3, 2, 4).contiguous().view(Bh, v.shape[2], self.d_head)
        
        # Attention: (Bh, n_tar, n_obs)
        scores = torch.matmul(q2, k2.transpose(-2, -1)) / (self.d_head ** 0.5)
        att = torch.softmax(scores, dim=-1)
        
        # Context: (Bh, n_tar, d_head)
        ctx2 = torch.matmul(att, v2)
        
        # Merge heads back: (Bh, n_tar, d_head) → (B, H, n_tar, d_total)
        ctxh = ctx2.view(B, H, self.n_heads, n_tar, self.d_head).permute(0, 1, 3, 2, 4)
        ctx = ctxh.contiguous().view(B, H, n_tar, self.d_total)
        
        # Output MLP
        y = self.mlp(ctx)  # (B, H, n_tar, 1)
        return y


class STNFXAttn(nn.Module):
    """STNF-XAttn 전체 모델 (원본 구조)"""
    
    def __init__(self, p: int = 0, m: int = 64, d_site: int = 128,
                 h_temporal: int = 128, dropout: float = 0.0, layernorm: bool = False,
                 n_heads: int = 3, use_basis_embedding: bool = True,
                 basis_k: int = 250, basis_initialize: Literal['regular'] = 'regular',
                 basis_learnable: bool = False):
        super().__init__()
        self.p = p
        self.use_basis_embedding = use_basis_embedding
        
        # Basis embedding - m 계산을 먼저 해야 함!
        if use_basis_embedding:
            self.basis_embedding = BasisEmbedding(
                k=basis_k,
                basis_initialize=basis_initialize,
                basis_learnable=basis_learnable
            )
            # m = k (basis function values)
            self.m = basis_k
        else:
            self.basis_embedding = None
            self.m = 2  # If no basis, use raw coords (2D)
        
        # Encode each observed site per time step
        self.encoder = SiteEncoder(self.p, self.m, d_site, dropout=dropout, layernorm=layernorm)
        
        # Temporal roll based on mean embedding over observed sites
        self.temporal = GRURoll(d_site, h=h_temporal, dropout=dropout)
        
        # Multi-head cross-attention from temporal state to observed-site embeddings
        self.head = CrossAttnHead(h_temporal, self.m, self.p, d=d_site, dropout=dropout,
                                  layernorm=layernorm, n_heads=n_heads)

    def forward(self, obs_coords: torch.Tensor, target_coords: torch.Tensor,
                y_hist_obs: torch.Tensor, H: int,
                X_hist_obs: Optional[torch.Tensor] = None,
                X_fut_target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        obs_coords: (B, n_obs, 2) or (n_obs, 2)
        target_coords: (B, n_tar, 2) or (n_tar, 2)
        y_hist_obs: (B, L, n_obs, 1)
        H: forecast horizon
        X_hist_obs: (B, L, n_obs, p) - optional covariates
        X_fut_target: (H, n_tar, p) or (n_tar, p) - optional
        
        Returns: y_pred: (B, H, n_tar, 1)
        """
        B, L, n_obs, _ = y_hist_obs.shape
        
        # Apply basis embedding
        if self.basis_embedding is not None:
            if obs_coords.dim() == 2:
                # (n_obs, 2) → (n_obs, m)
                phi_obs = self.basis_embedding(obs_coords.unsqueeze(0)).squeeze(0)
            else:
                # (B, n_obs, 2) → (B, n_obs, m)
                phi_obs = self.basis_embedding(obs_coords)
            
            if target_coords.dim() == 2:
                # (n_tar, 2) → (n_tar, m)
                phi_target = self.basis_embedding(target_coords.unsqueeze(0)).squeeze(0)
            else:
                # (B, n_tar, 2) → (B, n_tar, m)
                phi_target = self.basis_embedding(target_coords)
        else:
            phi_obs = obs_coords
            phi_target = target_coords
        
        # Broadcast phi_obs to (B, n_obs, m)
        if phi_obs.dim() == 2:
            phi_obs = phi_obs.unsqueeze(0).expand(B, -1, -1)
        
        # Prepare X_hist_obs (covariates)
        if X_hist_obs is None:
            # Create dummy covariates with zeros
            X_hist_obs = torch.zeros(B, L, n_obs, self.p, device=y_hist_obs.device)
        
        # Encode each time slice using observed y at that time with X at that time
        E_list = []
        for t in range(L):
            # X_hist_obs[:, t]: (B, n_obs, p)
            # phi_obs: (B, n_obs, m)
            # y_hist_obs[:, t]: (B, n_obs, 1)
            E_t = self.encoder(X_hist_obs[:, t], phi_obs, y_hist_obs[:, t])  # (B, n_obs, d)
            E_list.append(E_t)
        
        E_seq = torch.stack(E_list, dim=1)  # (B, L, n_obs, d)
        
        # Observed-site embeddings (use ALL time steps for temporal attention)
        # Reshape: (B, L, n_obs, d) → (B, L*n_obs, d)
        H_emb_obs = E_seq.view(B, L * n_obs, self.d)  # (B, L*n_obs, d)
        
        # Temporal roll from averaged embedding across sites
        E_mean = E_seq.mean(dim=2)  # (B, L, d)
        A = self.temporal(E_mean, H)  # (B, H, h)
        
        # Prepare X_target
        if phi_target.dim() == 2:
            n_tar = phi_target.shape[0]
        else:
            n_tar = phi_target.shape[1]
        
        if X_fut_target is None:
            X_target = torch.zeros(n_tar, self.p, device=y_hist_obs.device)
        else:
            X_target = X_fut_target
        
        # Cross-attend temporal states to observed-site embeddings to predict at targets
        y_hat = self.head(A, phi_target, X_target, H_emb_obs)  # (B, H, n_tar, 1)
        
        return y_hat


def create_model(config: dict) -> STNFXAttn:
    """설정에서 모델 생성"""
    model = STNFXAttn(
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
