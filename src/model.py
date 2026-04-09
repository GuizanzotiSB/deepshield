"""Arquitetura do modelo DeepShield.

Define :class:`DeepShieldModel`, baseado em EfficientNet-B0 (via ``timm``),
com cabeça de classificação customizada e utilitários para transfer
learning em duas fases.

Estratégia de transfer learning:

* **Fase 1** – ``freeze_backbone()`` + treinar apenas o classificador (5 épocas).
* **Fase 2** – ``unfreeze_last_blocks(n=3)`` + fine-tuning com LR menor (10 épocas).
"""
from __future__ import annotations

from typing import Iterable

import timm
import torch
import torch.nn as nn


class DeepShieldModel(nn.Module):
    """Detector de deepfakes baseado em EfficientNet-B0."""

    BACKBONE_NAME: str = "efficientnet_b0"
    FEATURE_DIM: int = 1280  # saída do EfficientNet-B0 (pool global)

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        """Inicializa o modelo.

        Args:
            num_classes: Número de classes de saída (2 = real/fake).
            pretrained: Se True, carrega pesos pré-treinados no ImageNet.
            dropout: Probabilidade de dropout na cabeça de classificação.
        """
        super().__init__()

        # num_classes=0 remove a cabeça original -> saída de features (1280,)
        self.backbone: nn.Module = timm.create_model(
            self.BACKBONE_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        self.classifier: nn.Sequential = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    # ------------------------------------------------------------------ #
    # Forward / features
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa o forward pass completo."""
        features = self.backbone(x)
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna o último feature map espacial do backbone.

        Útil para visualizações como Grad-CAM. Mantém as dimensões
        espaciais (antes do pool global).
        """
        return self.backbone.forward_features(x)

    # ------------------------------------------------------------------ #
    # Transfer learning helpers
    # ------------------------------------------------------------------ #
    def freeze_backbone(self) -> None:
        """Congela todos os parâmetros do backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Descongela todos os parâmetros do backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_blocks(self, n: int = 3) -> None:
        """Descongela os últimos ``n`` blocos do EfficientNet.

        O EfficientNet-B0 do ``timm`` expõe ``backbone.blocks`` como uma
        ``nn.Sequential`` de estágios. Também descongela as camadas finais
        (``conv_head`` e ``bn2``) para permitir fine-tuning adequado.
        """
        self.freeze_backbone()

        blocks: Iterable[nn.Module] = getattr(self.backbone, "blocks", [])
        blocks_list = list(blocks)
        for block in blocks_list[-n:]:
            for param in block.parameters():
                param.requires_grad = True

        for attr in ("conv_head", "bn2"):
            module = getattr(self.backbone, attr, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True

    # ------------------------------------------------------------------ #
    # Inspeção
    # ------------------------------------------------------------------ #
    def count_parameters(self) -> dict[str, int]:
        """Conta parâmetros treináveis vs. congelados.

        Returns:
            Dicionário com chaves ``trainable``, ``frozen`` e ``total``.
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}

    def summary(self) -> None:
        """Imprime um resumo da arquitetura e da contagem de parâmetros."""
        counts = self.count_parameters()
        print("=" * 60)
        print("DeepShieldModel - EfficientNet-B0 + classificador customizado")
        print("=" * 60)
        print(f"Backbone        : {self.BACKBONE_NAME}")
        print(f"Feature dim     : {self.FEATURE_DIM}")
        print("Classifier      : Linear(1280,512) -> ReLU -> Dropout(0.3) -> Linear(512,2)")
        print("-" * 60)
        print(f"Parâmetros totais   : {counts['total']:,}")
        print(f"Treináveis          : {counts['trainable']:,}")
        print(f"Congelados          : {counts['frozen']:,}")
        print("=" * 60)


if __name__ == "__main__":
    model = DeepShieldModel(pretrained=False)
    model.summary()

    print("\n[Fase 1] freeze_backbone()")
    model.freeze_backbone()
    print(model.count_parameters())

    print("\n[Fase 2] unfreeze_last_blocks(n=3)")
    model.unfreeze_last_blocks(n=3)
    print(model.count_parameters())

    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    feats = model.extract_features(dummy)
    print(f"\nforward output shape : {tuple(out.shape)}")
    print(f"feature map shape    : {tuple(feats.shape)}")
