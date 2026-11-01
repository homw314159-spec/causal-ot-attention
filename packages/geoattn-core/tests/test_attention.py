import torch
import torch.nn.functional as F

from geoattn_core import (
    AttentionProbe,
    GeometricAttention,
    IdentityPE,
    Normalizer,
    PositionEncoding,
    Sinkhorn,
    Softmax,
)


def test_implementations_satisfy_protocols() -> None:
    assert isinstance(Softmax(), Normalizer)
    assert isinstance(Sinkhorn(), Normalizer)
    assert isinstance(IdentityPE(), PositionEncoding)


def test_softmax_attention_matches_torch_sdpa() -> None:
    torch.manual_seed(0)
    b, n, d, h = 2, 7, 32, 4
    attn = GeometricAttention(d, h, pe=IdentityPE(), normalizer=Softmax()).eval()
    x = torch.randn(b, n, d)

    q, k, v = attn.qkv(x).reshape(b, n, 3, h, d // h).permute(2, 0, 3, 1, 4).unbind(0)
    ref = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
    ref = attn.proj(ref)

    torch.testing.assert_close(attn(x), ref, atol=1e-5, rtol=1e-4)


def test_probe_sees_coupling() -> None:
    torch.manual_seed(0)
    attn = GeometricAttention(16, 2, pe=IdentityPE(), normalizer=Sinkhorn(n_iters=50)).eval()
    seen: list[torch.Tensor] = []

    def hook(_m: torch.nn.Module, _i: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        seen.append(out)

    attn.probe.register_forward_hook(hook)
    attn(torch.randn(3, 5, 16))

    (p,) = seen
    assert isinstance(attn.probe, AttentionProbe)
    assert p.shape == (3, 2, 5, 5)
    torch.testing.assert_close(p.sum(-2), torch.ones(3, 2, 5), atol=1e-4, rtol=0)


def test_module_apply_still_walks_the_tree() -> None:
    """Regression: the PE hook must not shadow nn.Module.apply (ADR-0002)."""
    attn = GeometricAttention(16, 2, pe=IdentityPE(), normalizer=Softmax())
    visited: list[str] = []
    attn.apply(lambda m: visited.append(type(m).__name__))
    assert "IdentityPE" in visited and "Softmax" in visited


def test_decoupled_head_dim() -> None:
    """``head_dim`` independent of ``dim`` (Sinkformers MNIST: dim 128, 1 head of 64)."""
    attn = GeometricAttention(
        128, 1, pe=IdentityPE(), normalizer=Sinkhorn(n_iters=3), qkv_bias=False, head_dim=64
    )
    assert attn.qkv.weight.shape == (3 * 64, 128) and attn.qkv.bias is None
    assert attn.proj.weight.shape == (128, 64)
    out = attn(torch.randn(2, 5, 128))
    assert out.shape == (2, 5, 128)
    out.sum().backward()
    assert all(p.grad is not None for p in attn.parameters())


def test_backward_runs_with_sinkhorn() -> None:
    attn = GeometricAttention(16, 2, pe=IdentityPE(), normalizer=Sinkhorn(n_iters=3))
    attn(torch.randn(2, 5, 16)).sum().backward()
    assert all(p.grad is not None for p in attn.parameters())
