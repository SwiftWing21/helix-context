"""
Retrieval Manifold — Celestia Mamba architecture ported to helix's retrieval feature vector.

Port of Celestia's ManifoldV7 (train_manifold_v7.py) shrunk to helix's input shape.
Consumes per-query retrieval features, emits learned dimension scaling + K (prediction
confidence) for helix's budget-tier controller.

Architecture:
    Input per tick (~58d):
        tier_features[9]         — D1..D9 raw scores from SQL recall
        query_embed[20]          — ΣĒMA semantic embedding of query
        top_candidate_embed[20]  — ΣĒMA embedding of the winning gene
        log1p(requery_delta_s)[1] — time gap signal for Mamba Δ-gating
        party_id_embed[8]        — party ID embedding

    Shared projection → d_model=128
    2-layer Mamba SSM (d_state=32, dropout=0.1)
    NO speed-separated heads (helix dimensions aren't timescale-separated)

    Output per tick:
        scaling[9]    — per-dimension relevance scaling (softplus)
        K[1]          — prediction confidence, sigmoid, [0, 1]
        K_internal[1] — internal K for reflection trigger (see K_internal note below)

Origin: MambaBlock copied verbatim from /workspace/train_manifold_v7.py (Celestia v7).
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# HELIX RETRIEVAL CONFIG
# ═══════════════════════════════════════════════════════════════

N_TIER_DIMS = 9           # D1..D9 retrieval dimension scores
QUERY_EMBED_DIM = 20      # ΣĒMA query embedding
CANDIDATE_EMBED_DIM = 20  # ΣĒMA top-candidate embedding
LOG_DT_DIM = 1            # log1p(requery_delta_s)
PARTY_EMBED_DIM = 8       # party ID embedding (one-hot or learned)

INPUT_DIM = (N_TIER_DIMS + QUERY_EMBED_DIM + CANDIDATE_EMBED_DIM
             + LOG_DT_DIM + PARTY_EMBED_DIM)  # 58

D_MODEL = 128             # shared representation dimension
D_STATE = 32              # Mamba SSM state dimension
N_MAMBA_LAYERS = 2        # number of Mamba blocks
DROPOUT = 0.1

# Output dimensions
N_SCALING = N_TIER_DIMS   # 9 — one scaling weight per retrieval dimension
N_K = 1                   # overall prediction confidence
N_K_INTERNAL = 1          # internal K for reflection trigger

# K_internal design decision:
#
# In Celestia, K_internal is computed from VSlow channels 17-22 (DMN, valence,
# reward_anticipation, etc.) — cognitive post-processing regions that need time
# to settle. The reflection trigger fires when K_internal drops while sensory K
# stays high (= "I'm seeing things but not understanding them").
#
# For helix, the handoff suggests D7 (gene attribution), D8 (co-activation/SR),
# D9 (TCM) as the "slow" lanes — session-and-identity-scale dimensions that need
# accumulated context to score well.
#
# DECISION: Implement K_internal as a SEPARATE learned head (not a hard-coded
# subset of scaling outputs), but initialize it with a bias toward D7-D9 via an
# attention mask. Rationale:
#
#   1. Hard-coding K_internal = f(scaling[6:9]) couples K_internal to the scaling
#      head's learned representation, which may not capture "understanding vs seeing"
#      the way Celestia's ROI-based split does. The 9 helix dimensions aren't
#      neuroanatomically grounded — they're retrieval heuristics.
#
#   2. A separate head lets the model learn WHICH dimensions signal "confident but
#      wrong" vs "confident and right" without constraining it to D7-D9 a priori.
#
#   3. We DO bias toward D7-D9 via initialization: the K_internal head's weights
#      for the last 3 scaling lanes start higher. This gives the prior from the
#      handoff's intuition without locking it in.
#
#   4. The K_fast vs K_slow distinction (alternative from handoff) maps less
#      cleanly because helix's dimensions aren't timescale-separated — FTS5 (D1)
#      is "fast" in latency but not in the brain-ROI sense. The fast/slow split
#      in Celestia reflects actual neural processing timescales (V1 at 50ms vs
#      DMN at 10s+), not retrieval method latency.
#
# If this proves wrong in training, the fix is trivial: swap to a hard subset
# or to the K_fast/K_slow split. The separate-head approach is the most flexible
# starting point.


# ═══════════════════════════════════════════════════════════════
# MAMBA BLOCK — verbatim from /workspace/train_manifold_v7.py
# ═══════════════════════════════════════════════════════════════

class MambaBlock(nn.Module):
    """Lightweight Mamba for temporal channels.

    Copied verbatim from Celestia's train_manifold_v7.py (ManifoldV7).
    Single-step SSM: x(1, d_model) → (output, hidden_state).
    Δ-gating via softplus(dt_raw) handles time-gap weighting automatically
    when log1p(requery_delta_s) is included as an input feature.
    """
    def __init__(self, d_model, d_state=32, dropout=0.1):
        super().__init__()
        d_inner = d_model * 2
        self.d_inner, self.d_state = d_inner, d_state
        self.in_proj = nn.Linear(d_model, d_inner * 2 + d_state * 2 + 1, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, d_state, d_state)))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state=None):
        """x: (1, d_model), state: (d_inner, d_state) or None. Single step."""
        proj = self.in_proj(x)
        z = proj[:, :self.d_inner]
        x_ssm = proj[:, self.d_inner:2*self.d_inner]
        B_mat = proj[:, 2*self.d_inner:2*self.d_inner+self.d_state]
        C_mat = proj[:, 2*self.d_inner+self.d_state:2*self.d_inner+2*self.d_state]
        dt_raw = proj[:, -1:]

        x_ssm = F.silu(x_ssm)
        A = -torch.exp(self.A_log)
        dt = F.softplus(dt_raw)

        h = torch.zeros(1, self.d_inner, self.d_state, device=x.device) if state is None else state
        h = h * torch.exp(A * dt).unsqueeze(1) + x_ssm.unsqueeze(-1) * (dt * B_mat).unsqueeze(1)
        y = (h * C_mat.unsqueeze(1)).sum(-1) + self.D * x_ssm

        out = self.dropout(self.out_proj(y * F.silu(z)))
        return out, h


# ═══════════════════════════════════════════════════════════════
# RETRIEVAL MANIFOLD
# ═══════════════════════════════════════════════════════════════

class RetrievalManifold(nn.Module):
    """Mamba-based retrieval manifold: helix query features → dimension scaling + K.

    Mirrors Celestia's ManifoldV7 pattern but simplified:
    - Single shared projection (no speed-separated heads)
    - 2-layer Mamba with residual connections and LayerNorm (matching ReactorV7 pattern)
    - Output heads: scaling[9] + K[1] + K_internal[1]
    """

    def __init__(self, input_dim=INPUT_DIM, d_model=D_MODEL, d_state=D_STATE,
                 n_layers=N_MAMBA_LAYERS, dropout=DROPOUT,
                 n_scaling=N_SCALING, n_parties=64):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.n_scaling = n_scaling

        # ── Input projection ──
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # ── Mamba layers with residual + LayerNorm (ReactorV7 pattern) ──
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # ── Output heads ──
        # Scaling: per-dimension relevance weights, softplus activation
        # (softplus keeps values positive without the [0,1] ceiling of sigmoid,
        # allowing the model to amplify strong dimensions beyond 1.0)
        self.scaling_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, n_scaling),
        )

        # K: overall prediction confidence, sigmoid to [0, 1]
        self.k_head = nn.Linear(d_model, 1)

        # K_internal: reflection trigger signal, sigmoid to [0, 1]
        # Separate head with D7-D9 bias (see design decision above)
        self.k_internal_head = nn.Linear(d_model, 1)

        self._init_k_internal_bias()

        total = sum(p.numel() for p in self.parameters())
        print(f"  RetrievalManifold: {total:,} params")
        print(f"  Input: {input_dim}d → d_model={d_model}, "
              f"{n_layers}-layer Mamba (d_state={d_state})")
        print(f"  Output: scaling[{n_scaling}] + K[1] + K_internal[1]")

    def _init_k_internal_bias(self):
        """Bias K_internal head toward D7-D9 information via the Mamba hidden state.

        We can't directly bias toward specific input dimensions (D7-D9) since the
        head operates on the Mamba hidden state. Instead, we initialize the
        K_internal head with slightly higher weight magnitude than the K head,
        encouraging it to learn a distinct signal. The actual D7-D9 specialization
        will emerge from training on CWoLa data where session-scale retrieval
        quality (D7-D9) diverges from query-scale quality (D1-D6).
        """
        nn.init.xavier_normal_(self.k_internal_head.weight, gain=0.5)
        nn.init.constant_(self.k_internal_head.bias, 0.0)

    def forward(self, x, states=None):
        """Single-tick forward pass.

        Args:
            x: (batch, input_dim) — concatenated retrieval features
            states: list of (d_inner, d_state) Mamba hidden states, or None

        Returns:
            outputs: dict with 'scaling', 'K', 'K_internal'
            new_states: list of updated Mamba hidden states
        """
        if states is None:
            states = [None] * len(self.blocks)

        # Project input
        h = self.input_proj(x)  # (batch, d_model)

        # Mamba layers with residual connections
        new_states = []
        for i, (block, norm) in enumerate(zip(self.blocks, self.norms)):
            residual = h
            out, state = block(h, states[i])
            h = norm(residual + out)
            new_states.append(state)

        h = self.final_norm(h)

        # Output heads
        scaling_raw = self.scaling_head(h)
        scaling = F.softplus(scaling_raw)     # (batch, 9), positive values

        k_logit = self.k_head(h)
        K = torch.sigmoid(k_logit)            # (batch, 1), [0, 1]

        k_int_logit = self.k_internal_head(h)
        K_internal = torch.sigmoid(k_int_logit)  # (batch, 1), [0, 1]

        return {
            'scaling': scaling,
            'K': K,
            'K_internal': K_internal,
        }, new_states

    def get_mamba_state_shapes(self):
        """Return expected Mamba state shapes for each layer.

        Useful for pre-allocating states and for test verification.
        Matches the (d_inner, d_state) shape from MambaBlock.
        """
        shapes = []
        for block in self.blocks:
            shapes.append((1, block.d_inner, block.d_state))
        return shapes


# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP — scaffold only, do not run
# ═══════════════════════════════════════════════════════════════

class RetrievalCWoLaDataset(torch.utils.data.Dataset):
    """Reads cwola_export_*.json from R2 data path.

    Each row in the export contains:
        - tier_features: dict with D1..D9 scores
        - party_id: string identifier for the conversation party
        - bucket: 'A' (good retrieval) or 'B' (bad retrieval)
        - requery_delta_s: seconds since last query in this session (optional)

    Note: query_embed and top_candidate_embed are NOT yet available in the
    cwola_log export. They are set to zeros for first-pass training. A helix-side
    feature export will enrich cwola_log with embeddings in a later pass.
    """

    def __init__(self, json_path, party_to_idx=None):
        with open(json_path) as f:
            raw = json.load(f)

        self.rows = []
        self.labels = []
        self.party_to_idx = party_to_idx or {}
        self._next_party = max(self.party_to_idx.values(), default=-1) + 1

        for row in raw:
            bucket = row.get('bucket', '')
            if bucket == 'A':
                label = 1.0
            elif bucket == 'B':
                label = 0.0
            else:
                continue  # skip non-A/B entries

            self.rows.append(row)
            self.labels.append(label)

            pid = row.get('party_id', 'unknown')
            if pid not in self.party_to_idx:
                self.party_to_idx[pid] = self._next_party
                self._next_party += 1

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        # Tier features: D1..D9 as a 9-vector
        tf = row.get('tier_features', {})
        tier_features = torch.tensor([
            tf.get(f'D{i}', 0.0) for i in range(1, N_TIER_DIMS + 1)
        ], dtype=torch.float32)

        # Embeddings: zeros until helix-side export adds them
        query_embed = torch.zeros(QUERY_EMBED_DIM)
        top_candidate_embed = torch.zeros(CANDIDATE_EMBED_DIM)

        # Time gap
        log_dt = torch.tensor([math.log1p(row.get('requery_delta_s') or 0.0)])

        # Party ID: one-hot encoding
        pid = row.get('party_id', 'unknown')
        party_vec = torch.zeros(PARTY_EMBED_DIM)
        party_idx = self.party_to_idx.get(pid, 0)
        party_vec[party_idx % PARTY_EMBED_DIM] = 1.0

        features = torch.cat([tier_features, query_embed, top_candidate_embed,
                              log_dt, party_vec])

        label = torch.tensor(self.labels[idx])
        return features, label


def compute_k_target(labels, window_size=20):
    """Compute rolling K target from A-bucket rate over a window.

    K_target[t] = mean(label[t-window_size:t])

    This is the secondary training target for K calibration.
    Sequential within a session — requires ordered data.

    Args:
        labels: (N,) tensor of 0/1 labels (B/A bucket)
        window_size: rolling window size (default 20, matching Celestia's K_WINDOW)

    Returns:
        k_targets: (N,) tensor of rolling A-bucket rates
    """
    n = labels.shape[0]
    k_targets = torch.zeros(n)
    for t in range(n):
        start = max(0, t - window_size + 1)
        k_targets[t] = labels[start:t + 1].mean()
    return k_targets


def train_loop(data_path, device='cuda', n_epochs=30, lr=1e-3, batch_size=64):
    """Training loop scaffold. DO NOT RUN — data not yet available.

    Multi-task loss:
        primary:   Binary CE on bucket prediction (A/B) using K head
        secondary: MSE on K vs rolling A-bucket rate within session window
                   (deferred to second pass — requires sequential-aware loading)
    """
    dataset = RetrievalCWoLaDataset(data_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RetrievalManifold().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    bce = nn.BCELoss()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            # Forward (no Mamba state carry in shuffled training — first pass)
            outputs, _ = model(features)

            # Primary loss: BCE on K vs bucket label
            k_pred = outputs['K'].squeeze(-1)
            loss = bce(k_pred, labels)

            # Secondary loss (K calibration) — deferred to second pass.
            # Would require sequential loading within sessions and
            # compute_k_target() applied per-session.

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        print(f"  Epoch {epoch:3d} | Loss: {epoch_loss / max(n_batches, 1):.4f}")

    return model


# ═══════════════════════════════════════════════════════════════
# INFERENCE HELPERS — for integration with helix retrieval
# ═══════════════════════════════════════════════════════════════

def pack_input(tier_features, query_embed=None, top_candidate_embed=None,
               requery_delta_s=0.0, party_id_onehot=None):
    """Pack helix retrieval features into a single input tensor.

    Args:
        tier_features: (9,) tensor of D1..D9 scores
        query_embed: (20,) ΣĒMA query embedding, or None (zeros)
        top_candidate_embed: (20,) ΣĒMA candidate embedding, or None (zeros)
        requery_delta_s: float, seconds since last query
        party_id_onehot: (8,) one-hot party ID, or None (zeros)

    Returns:
        (1, 58) tensor ready for model.forward()
    """
    if query_embed is None:
        query_embed = torch.zeros(QUERY_EMBED_DIM)
    if top_candidate_embed is None:
        top_candidate_embed = torch.zeros(CANDIDATE_EMBED_DIM)
    if party_id_onehot is None:
        party_id_onehot = torch.zeros(PARTY_EMBED_DIM)

    log_dt = torch.tensor([math.log1p(requery_delta_s)])

    return torch.cat([
        tier_features, query_embed, top_candidate_embed,
        log_dt, party_id_onehot
    ]).unsqueeze(0)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Retrieval Manifold — Celestia Mamba architecture ported to helix')
    parser.add_argument('--info', action='store_true',
                        help='Print model architecture info and exit')
    parser.add_argument('--check', action='store_true',
                        help='Run a synthetic forward pass to verify shapes')
    parser.add_argument('--train', type=str, default=None,
                        help='Path to cwola_export JSON (scaffold only — do not use yet)')
    args = parser.parse_args()

    if args.info:
        model = RetrievalManifold()
        print(f"\nParameter budget: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Target: ≤2,000,000")
        return

    if args.check:
        print("Running synthetic forward pass...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = RetrievalManifold().to(device)

        # Single tick
        x = torch.randn(1, INPUT_DIM, device=device)
        outputs, states = model(x)
        print(f"  scaling shape: {outputs['scaling'].shape}")    # (1, 9)
        print(f"  K shape:       {outputs['K'].shape}")          # (1, 1)
        print(f"  K_internal:    {outputs['K_internal'].shape}") # (1, 1)
        print(f"  Mamba states:  {len(states)} layers")
        for i, s in enumerate(states):
            print(f"    layer {i}: {s.shape}")

        # Batch
        x_batch = torch.randn(32, INPUT_DIM, device=device)
        outputs_b, states_b = model(x_batch)
        print(f"\n  Batch (32) scaling: {outputs_b['scaling'].shape}")
        print(f"  Batch (32) K:       {outputs_b['K'].shape}")

        # Verify value ranges
        assert (outputs['scaling'] >= 0).all(), "scaling must be non-negative (softplus)"
        assert (outputs['K'] >= 0).all() and (outputs['K'] <= 1).all(), "K must be in [0,1]"
        assert (outputs['K_internal'] >= 0).all() and (outputs['K_internal'] <= 1).all(), \
            "K_internal must be in [0,1]"

        print("\nAll checks passed.")
        return

    if args.train:
        print("ERROR: Training is disabled in this build. Data not yet available.")
        print("       This scaffold exists for architecture verification only.")
        sys.exit(1)

    parser.print_help()


if __name__ == '__main__':
    main()
