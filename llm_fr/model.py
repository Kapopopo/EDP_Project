"""Transformer causal compact : embeddings, attention, MLP et prédiction suivante."""

import torch  # Fournit tenseurs, gradients et accélération matérielle.
from torch import nn  # Fournit les couches entraînables.
from torch.nn import functional as F  # Fournit attention fusionnée et entropie croisée.


class Block(nn.Module):  # Regroupe attention causale et réseau de transformation.
    def __init__(self, width, heads, dropout):  # Reçoit dimension, nombre de têtes et régularisation.
        super().__init__()  # Enregistre correctement les sous-couches PyTorch.
        self.heads = heads  # Mémorise le nombre de vues d'attention.
        self.dropout = dropout  # Mémorise la probabilité de désactivation.
        self.norm1 = nn.LayerNorm(width)  # Normalise avant l'attention pour stabiliser l'apprentissage.
        self.qkv = nn.Linear(width, 3 * width, bias=False)  # Produit requêtes, clés et valeurs simultanément.
        self.proj = nn.Linear(width, width, bias=False)  # Recombine les têtes d'attention.
        self.norm2 = nn.LayerNorm(width)  # Normalise avant le réseau dense.
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width), nn.Dropout(dropout))  # Transforme chaque position indépendamment.
        self.residual_dropout = nn.Dropout(dropout)  # Régularise la sortie d'attention.

    def forward(self, x):  # Transforme un lot de séquences.
        batch, length, width = x.shape  # Récupère les dimensions du tenseur.
        qkv = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, width // self.heads)  # Sépare les trois projections et les têtes.
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # Place les axes comme attendu par l'attention.
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)  # Interdit de regarder les tokens futurs.
        attention = attention.transpose(1, 2).contiguous().view(batch, length, width)  # Rassemble les têtes.
        x = x + self.residual_dropout(self.proj(attention))  # Ajoute la connexion résiduelle d'attention.
        return x + self.mlp(self.norm2(x))  # Ajoute la transformation dense sans perdre l'entrée.


class LanguageModel(nn.Module):  # Définit le modèle autoregressif complet.
    def __init__(self, vocab_size, context, width, heads, layers, dropout):  # Reçoit uniquement des paramètres d'architecture.
        super().__init__()  # Initialise le suivi des paramètres.
        if min(vocab_size, context, width, heads, layers) <= 0 or width % heads or not 0 <= dropout < 1:  # Vérifie la géométrie des têtes.
            raise ValueError("Architecture invalide : dimensions positives, width divisible par heads, dropout dans [0,1[.")  # Signale le problème.
        self.context = context  # Limite la longueur des séquences.
        self.tokens = nn.Embedding(vocab_size, width)  # Apprend une représentation de chaque token.
        self.positions = nn.Embedding(context, width)  # Apprend l'ordre des positions.
        self.blocks = nn.Sequential(*(Block(width, heads, dropout) for _ in range(layers)))  # Empile les transformations.
        self.norm = nn.LayerNorm(width)  # Normalise la sortie finale.
        self.apply(self._initialize)  # Initialise tous les poids depuis zéro.

    @staticmethod  # L'initialisation n'utilise pas d'état propre au modèle.
    def _initialize(module):  # Applique une faible variance aux poids.
        if isinstance(module, (nn.Linear, nn.Embedding)):  # Sélectionne les couches à poids denses.
            nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Tire des poids aléatoires centrés.
            if isinstance(module, nn.Linear) and module.bias is not None:  # Repère les biais existants.
                nn.init.zeros_(module.bias)  # Démarre les biais à zéro.

    def forward(self, ids, targets=None):  # Calcule les scores et éventuellement la perte.
        if ids.shape[1] > self.context:  # Vérifie la capacité de positionnement.
            raise ValueError("Séquence plus longue que le contexte du modèle.")  # Évite un accès hors limites.
        positions = torch.arange(ids.shape[1], device=ids.device)  # Numérote les positions sur le bon matériel.
        x = self.tokens(ids) + self.positions(positions)  # Combine sens du token et position.
        x = self.norm(self.blocks(x))  # Applique tous les blocs puis la normalisation.
        logits = F.linear(x, self.tokens.weight)  # Partage les poids entrée/sortie pour économiser des paramètres.
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))  # Mesure l'erreur de prédiction du prochain token.
        return logits, loss  # Renvoie les scores et la perte moyenne.
