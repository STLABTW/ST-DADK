"""
STNF-XAttn: Spatio-Temporal Neural Field with Cross-Attention (v2)

사용자 제시 구조대로 재구현:
1. SiteEncoder: 각 시점별로 [φ, y_t, X_t] → E_seq (B, L, n_obs, d)
2. GRURoll: mean_sites(E_seq) → GRU → rollout A (B, H, h)
3. CrossAttnHead: Query=[A, φ_tar, X_tar], KV=H_emb,obs → y_pred (B, H, n_tar, 1)

Covariates X는 p=0으로 처리하되 확장 가능하도록 구조화
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal

from .basis_embedding import BasisEmbedding


class SiteEncoder(nn.Module):
    """
    시공간 특징을 인코딩 (시점별 처리)
    
    입력 (한 배치의 모든 시점):
        - phi_obs: (n_obs, m) 또는 (B, n_obs, m) - basis embedding
        - y_hist: (B, L, n_obs, 1) - 관측값
        - X_hist: (B, L, n_obs, p) - covariates (optional, p=0 가능)
    
    처리:
        각 시점 t=1..L에 대해:
        - concat([φ, y_t, X_t]) → (B, n_obs, m+1+p)
        - MLP → (B, n_obs, d)
        - 시간축 스택 → E_seq (B, L, n_obs, d)
    
    출력:
        - E_seq: (B, L, n_obs, d)
    """
    def __init__(self, d_site: int = 128, p_covariates: int = 0, dropout: float = 0.1):
        super().__init__()
        self.d_site = d_site
        self.p_covariates = p_covariates
        
        # Note: m (basis 차원)은 forward에서 결정됨
        # 일단 placeholder로 설정하고, forward에서 동적 처리
        # 실제로는 basis_k를 알아야 하지만, 나중에 수정 가능
        self.mlp = None  # forward에서 lazy initialization
        self.dropout = dropout
    
    def _build_mlp(self, input_dim: int):
        """MLP 동적 생성"""
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.d_site),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_site, self.d_site),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        ).to(next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu')
    
    def forward(
        self, 
        phi_obs: torch.Tensor, 
        y_hist: torch.Tensor,
        X_hist: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            phi_obs: (n_obs, m) 또는 (B, n_obs, m)
            y_hist: (B, L, n_obs, 1)
            X_hist: (B, L, n_obs, p) - optional
        
        Returns:
            E_seq: (B, L, n_obs, d)
        """
        B, L, n_obs, _ = y_hist.shape
        
        # phi_obs broadcast to (B, L, n_obs, m)
        if phi_obs.dim() == 2:
            # (n_obs, m) → (B, L, n_obs, m)
            m = phi_obs.shape[-1]
            phi_expanded = phi_obs.unsqueeze(0).unsqueeze(0).expand(B, L, n_obs, m)
        else:
            # (B, n_obs, m) → (B, L, n_obs, m)
            m = phi_obs.shape[-1]
            phi_expanded = phi_obs.unsqueeze(1).expand(B, L, n_obs, m)
        
        # Concatenate inputs
        if X_hist is not None and self.p_covariates > 0:
            # (B, L, n_obs, m+1+p)
            mlp_input = torch.cat([phi_expanded, y_hist, X_hist], dim=-1)
            input_dim = m + 1 + self.p_covariates
        else:
            # (B, L, n_obs, m+1)
            mlp_input = torch.cat([phi_expanded, y_hist], dim=-1)
            input_dim = m + 1
        
        # Lazy MLP initialization
        if self.mlp is None:
            self._build_mlp(input_dim)
        
        # Process: (B, L, n_obs, input_dim) → (B, L, n_obs, d)
        # Flatten for efficient processing
        mlp_input_flat = mlp_input.view(B * L * n_obs, input_dim)
        E_flat = self.mlp(mlp_input_flat)  # (B*L*n_obs, d)
        E_seq = E_flat.view(B, L, n_obs, self.d_site)
        
        return E_seq


class GRURoll(nn.Module):
    """
    GRU 기반 시간 인코딩 및 롤아웃
    
    입력:
        - E_seq: (B, L, n_obs, d) - SiteEncoder 출력
    
    처리:
        1. 사이트 차원 평균: mean_sites(E_seq) → (B, L, d)
        2. GRU 인코딩: (B, L, d) → h_L: (B, h)
        3. 롤아웃: h_L을 초기값으로 H-step 반복
           - context = E_seq[:, -1, :, :].mean(1) (B, d) 고정
           - h^(r) = GRUCell(context, h^(r-1))
           - A = [h^(1), ..., h^(H)]: (B, H, h)
    
    출력:
        - A: (B, H, h) - 롤아웃 hidden states
        - H_emb_obs: (B, n_obs, d) - 마지막 시점 관측 임베딩 (K/V용)
    """
    def __init__(self, d_site: int = 128, h_temporal: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_site = d_site
        self.h_temporal = h_temporal
        
        # GRU encoder
        self.gru = nn.GRU(
            input_size=d_site,
            hidden_size=h_temporal,
            num_layers=1,
            batch_first=True
        )
        
        # GRU rollout cell
        self.gru_cell = nn.GRUCell(
            input_size=d_site,
            hidden_size=h_temporal
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, E_seq: torch.Tensor, H: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            E_seq: (B, L, n_obs, d)
            H: 예측 horizon
        
        Returns:
            A: (B, H, h) - 롤아웃 상태
            H_emb_obs: (B, n_obs, d) - 관측 임베딩
        """
        B, L, n_obs, d = E_seq.shape
        
        # 1. 사이트 평균: (B, L, n_obs, d) → (B, L, d)
        E_mean = E_seq.mean(dim=2)
        
        # 2. GRU 인코딩
        _, h_L = self.gru(E_mean)  # h_L: (1, B, h)
        h_L = h_L.squeeze(0)  # (B, h)
        
        # 3. 컨텍스트 벡터: 마지막 시점 평균
        context = E_seq[:, -1, :, :].mean(dim=1)  # (B, d)
        
        # 4. 롤아웃
        h_t = h_L
        A_list = []
        
        for r in range(H):
            h_t = self.gru_cell(context, h_t)  # (B, h)
            h_t = self.dropout(h_t)
            A_list.append(h_t)
        
        A = torch.stack(A_list, dim=1)  # (B, H, h)
        
        # 5. 관측 임베딩 (마지막 시점)
        H_emb_obs = E_seq[:, -1, :, :]  # (B, n_obs, d)
        
        return A, H_emb_obs


class CrossAttnHead(nn.Module):
    """
    Cross-Attention 기반 예측 헤드
    
    입력:
        - A: (B, H, h) - GRU 롤아웃 상태
        - phi_tar: (n_tar, m) - 타깃 사이트 basis embedding
        - X_tar: (n_tar, p) - 타깃 사이트 covariates (optional)
        - H_emb_obs: (B, n_obs, d) - 관측 임베딩
    
    처리:
        1. Query 구성:
           - A → (B, H, n_tar, h) 확장
           - phi_tar → (B, H, n_tar, m) 브로드캐스트
           - X_tar → (B, H, n_tar, p) 브로드캐스트 (optional)
           - concat → (B, H, n_tar, h+m+p)
           - proj_q → (B, H, n_tar, d_tot)
        
        2. Key/Value:
           - H_emb_obs: (B, n_obs, d) → proj_k/v → (B, n_obs, d_tot)
           - 허라이즌 차원으로 확장 → (B, H, n_obs, d_tot)
        
        3. Multi-head Attention
        
        4. FFN → (B, H, n_tar, 1)
    
    출력:
        - y_pred: (B, H, n_tar, 1)
    """
    def __init__(
        self, 
        h_temporal: int = 128,
        d_site: int = 128, 
        n_heads: int = 4, 
        p_covariates: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.h_temporal = h_temporal
        self.d_site = d_site
        self.n_heads = n_heads
        self.p_covariates = p_covariates
        
        # Note: m (basis 차원)은 forward에서 결정
        # d_tot는 h_temporal과 동일하게 설정 (간단하게)
        self.d_tot = h_temporal
        self.d_head = self.d_tot // n_heads
        
        assert self.d_tot % n_heads == 0, "d_tot must be divisible by n_heads"
        
        # Query projection (lazy initialization)
        self.proj_q = None
        
        # Key/Value projections
        self.proj_k = nn.Linear(d_site, self.d_tot)
        self.proj_v = nn.Linear(d_site, self.d_tot)
        
        # Output FFN
        self.output_ffn = nn.Sequential(
            nn.Linear(self.d_tot, self.d_tot),
            nn.LayerNorm(self.d_tot),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_tot, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _build_proj_q(self, input_dim: int):
        """Query projection 동적 생성"""
        self.proj_q = nn.Linear(input_dim, self.d_tot).to(
            next(self.parameters()).device
        )
    
    def forward(
        self,
        A: torch.Tensor,
        phi_tar: torch.Tensor,
        H_emb_obs: torch.Tensor,
        X_tar: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            A: (B, H, h)
            phi_tar: (n_tar, m) 또는 (B, n_tar, m)
            H_emb_obs: (B, n_obs, d)
            X_tar: (n_tar, p) 또는 (B, n_tar, p) - optional
        
        Returns:
            y_pred: (B, H, n_tar, 1)
        """
        B, H, h = A.shape
        n_obs, d = H_emb_obs.shape[1], H_emb_obs.shape[2]
        
        # Determine n_tar and m
        if phi_tar.dim() == 2:
            n_tar, m = phi_tar.shape
            # (n_tar, m) → (B, H, n_tar, m)
            phi_tar_exp = phi_tar.unsqueeze(0).unsqueeze(0).expand(B, H, n_tar, m)
        else:
            n_tar, m = phi_tar.shape[1], phi_tar.shape[2]
            # (B, n_tar, m) → (B, H, n_tar, m)
            phi_tar_exp = phi_tar.unsqueeze(1).expand(B, H, n_tar, m)
        
        # 1. Query 구성
        # A: (B, H, h) → (B, H, n_tar, h)
        A_exp = A.unsqueeze(2).expand(B, H, n_tar, h)
        
        # Concatenate
        if X_tar is not None and self.p_covariates > 0:
            # X_tar: (n_tar, p) → (B, H, n_tar, p)
            if X_tar.dim() == 2:
                X_tar_exp = X_tar.unsqueeze(0).unsqueeze(0).expand(B, H, n_tar, self.p_covariates)
            else:
                X_tar_exp = X_tar.unsqueeze(1).expand(B, H, n_tar, self.p_covariates)
            
            query_input = torch.cat([A_exp, phi_tar_exp, X_tar_exp], dim=-1)
            query_input_dim = h + m + self.p_covariates
        else:
            query_input = torch.cat([A_exp, phi_tar_exp], dim=-1)
            query_input_dim = h + m
        
        # Lazy proj_q initialization
        if self.proj_q is None:
            self._build_proj_q(query_input_dim)
        
        # Query projection: (B, H, n_tar, h+m+p) → (B, H, n_tar, d_tot)
        Q = self.proj_q(query_input)
        
        # 2. Key/Value
        # H_emb_obs: (B, n_obs, d) → (B, n_obs, d_tot)
        K = self.proj_k(H_emb_obs)
        V = self.proj_v(H_emb_obs)
        
        # 허라이즌 차원 확장: (B, n_obs, d_tot) → (B, H, n_obs, d_tot)
        K = K.unsqueeze(1).expand(B, H, n_obs, self.d_tot)
        V = V.unsqueeze(1).expand(B, H, n_obs, self.d_tot)
        
        # 3. Multi-head Attention
        # Reshape for multi-head: (B, H, n_tar/n_obs, d_tot) → (B, H, n_tar/n_obs, n_heads, d_head)
        Q = Q.view(B, H, n_tar, self.n_heads, self.d_head)
        K = K.view(B, H, n_obs, self.n_heads, self.d_head)
        V = V.view(B, H, n_obs, self.n_heads, self.d_head)
        
        # Transpose for attention: (B, H, n_heads, n_tar/n_obs, d_head)
        Q = Q.permute(0, 1, 3, 2, 4)  # (B, H, n_heads, n_tar, d_head)
        K = K.permute(0, 1, 3, 2, 4)  # (B, H, n_heads, n_obs, d_head)
        V = V.permute(0, 1, 3, 2, 4)  # (B, H, n_heads, n_obs, d_head)
        
        # Attention scores: (B, H, n_heads, n_tar, n_obs)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Context: (B, H, n_heads, n_tar, d_head)
        ctx = torch.matmul(attn_weights, V)
        
        # Merge heads: (B, H, n_heads, n_tar, d_head) → (B, H, n_tar, d_tot)
        ctx = ctx.permute(0, 1, 3, 2, 4).contiguous()
        ctx = ctx.view(B, H, n_tar, self.d_tot)
        
        # 4. Output FFN
        y_pred = self.output_ffn(ctx)  # (B, H, n_tar, 1)
        
        return y_pred


class STNFXAttn(nn.Module):
    """
    STNF-XAttn 전체 모델 (v2 - 사용자 구조대로)
    
    Args:
        d_site: SiteEncoder 출력 차원
        h_temporal: GRU hidden 차원
        n_heads: Cross-Attention head 수
        p_covariates: Covariates 차원 (기본 0)
        dropout: Dropout 비율
        use_basis_embedding: Basis embedding 사용 여부
        basis_k: Basis embedding 차원
        basis_initialize: Basis 초기화 방법
        basis_learnable: Basis 학습 여부
    """
    def __init__(
        self,
        d_site: int = 128,
        h_temporal: int = 128,
        n_heads: int = 4,
        p_covariates: int = 0,
        dropout: float = 0.1,
        use_basis_embedding: bool = True,
        basis_k: int = 250,
        basis_initialize: Literal['regular'] = 'regular',
        basis_learnable: bool = False
    ):
        super().__init__()
        self.d_site = d_site
        self.h_temporal = h_temporal
        self.use_basis_embedding = use_basis_embedding
        self.basis_k = basis_k
        
        # Basis Embedding
        if use_basis_embedding:
            self.basis_embedding = BasisEmbedding(
                k=basis_k,
                basis_initialize=basis_initialize,
                basis_learnable=basis_learnable
            )
        else:
            self.basis_embedding = None
        
        # Components
        self.site_encoder = SiteEncoder(d_site, p_covariates, dropout)
        self.gru_roll = GRURoll(d_site, h_temporal, dropout)
        self.cross_attn_head = CrossAttnHead(h_temporal, d_site, n_heads, p_covariates, dropout)
    
    def forward(
        self,
        obs_coords: torch.Tensor,
        target_coords: torch.Tensor,
        y_hist_obs: torch.Tensor,
        H: int,
        X_hist_obs: Optional[torch.Tensor] = None,
        X_fut_target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            obs_coords: (B, n_obs, 2) - 관측 좌표
            target_coords: (B, n_tar, 2) 또는 (n_tar, 2) - 타깃 좌표
            y_hist_obs: (B, L, n_obs, 1) - 관측 시계열
            H: 예측 horizon
            X_hist_obs: (B, L, n_obs, p) - 관측 covariates (optional)
            X_fut_target: (B, H, n_tar, p) 또는 (n_tar, p) - 타깃 covariates (optional)
        
        Returns:
            y_pred: (B, H, n_tar, 1)
        """
        # 1. Basis embedding
        if self.basis_embedding is not None:
            phi_obs = self.basis_embedding(obs_coords)  # (B, n_obs, m)
            
            if target_coords.dim() == 2:
                phi_tar = self.basis_embedding(target_coords)  # (n_tar, m)
            else:
                phi_tar = self.basis_embedding(target_coords)  # (B, n_tar, m)
        else:
            phi_obs = obs_coords  # (B, n_obs, 2)
            phi_tar = target_coords  # (B, n_tar, 2) or (n_tar, 2)
        
        # 2. SiteEncoder: [φ, y_t, X_t] → E_seq
        E_seq = self.site_encoder(phi_obs, y_hist_obs, X_hist_obs)  # (B, L, n_obs, d)
        
        # 3. GRURoll: mean_sites(E_seq) → A, H_emb_obs
        A, H_emb_obs = self.gru_roll(E_seq, H)  # A: (B, H, h), H_emb_obs: (B, n_obs, d)
        
        # 4. CrossAttnHead: [A, φ_tar, X_tar] → y_pred
        # Note: X_fut_target는 현재 시점에서는 미래의 첫 시점만 사용 (간단하게)
        # 실제로는 각 horizon마다 다른 X를 사용할 수도 있음
        X_tar_input = None
        if X_fut_target is not None:
            # (B, H, n_tar, p) → (B, n_tar, p) - 첫 시점만 사용
            if X_fut_target.dim() == 4:
                X_tar_input = X_fut_target[:, 0, :, :]
            else:
                X_tar_input = X_fut_target
        
        y_pred = self.cross_attn_head(A, phi_tar, H_emb_obs, X_tar_input)  # (B, H, n_tar, 1)
        
        return y_pred


def create_model(config: dict) -> STNFXAttn:
    """
    설정에서 모델 생성
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        model: STNFXAttn 인스턴스
    """
    model = STNFXAttn(
        d_site=config.get('d_site', 128),
        h_temporal=config.get('h_temporal', 128),
        n_heads=config.get('n_heads', 4),
        p_covariates=config.get('p_covariates', 0),
        dropout=config.get('dropout', 0.1),
        use_basis_embedding=config.get('use_basis_embedding', True),
        basis_k=config.get('basis_k', 250),
        basis_initialize=config.get('basis_initialize', 'regular'),
        basis_learnable=config.get('basis_learnable', False)
    )
    return model
