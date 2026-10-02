"""Vérifie le décodage progressif des accents et l'arrêt sur fin de texte."""

import unittest  # Organise les vérifications automatiques.
import torch  # Construit des scores de tokens déterministes.
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders  # Fabrique un tokenizer à octets miniature.
from llm_fr.chat import stream_reply, visible  # Teste le flux réellement affiché dans le terminal.


class ChatTests(unittest.TestCase):  # Vérifie les comportements spécifiques au mode direct.
    def test_utf8_stream_and_eos(self):  # Vérifie les accents répartis sur plusieurs tokens.
        tokenizer = Tokenizer(models.BPE())  # Initialise un BPE sans vocabulaire préexistant.
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # Encode tous les octets UTF-8.
        tokenizer.decoder = decoders.ByteLevel()  # Reconstruit les caractères accentués.
        trainer = trainers.BpeTrainer(vocab_size=257, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)  # Évite les fusions pour forcer plusieurs octets par accent.
        tokenizer.train_from_iterator(["été"], trainer=trainer)  # Entraîne le tokenizer minimal.
        expected = "été\n"  # Définit le texte attendu, avec accents et saut de ligne.
        sequence = iter(tokenizer.encode(expected).ids + [tokenizer.token_to_id("<|endoftext|>")])  # Termine la séquence par EOS.

        class FixedModel:  # Remplace uniquement le tirage du réseau par des scores déterministes.
            context = 8  # Fournit un petit contexte accepté par le générateur.

            def __call__(self, ids):  # Produit le prochain token prévu par le scénario.
                scores = torch.zeros(1, ids.shape[1], tokenizer.get_vocab_size())  # Crée les scores de chaque position.
                scores[:, -1, next(sequence)] = 100  # Force le prochain octet ou EOS.
                return scores, None  # Respecte l'interface du véritable modèle.

        pieces = list(stream_reply(FixedModel(), tokenizer, "Bonjour", 50, 1.0, 1))  # Exécute le véritable décodage progressif jusqu'à EOS.
        self.assertEqual("".join(pieces), expected)  # Exige un texte identique sans accents cassés.
        self.assertGreater(len(pieces), 1)  # Vérifie qu'il existe plusieurs fragments de sortie.
        self.assertTrue(all("\ufffd" not in piece for piece in pieces))  # Aucun octet incomplet ne doit apparaître à l'écran.

    def test_terminal_controls(self):  # Vérifie la neutralisation des caractères de contrôle.
        self.assertEqual(visible("Bonjour\x1b\x00\r\nété\t!"), "Bonjour\nété\t!")  # Conserve accents et espaces utiles, retire ESC et NUL.


if __name__ == "__main__":  # Autorise le lancement autonome des tests.
    unittest.main()  # Exécute les tests déclarés.
