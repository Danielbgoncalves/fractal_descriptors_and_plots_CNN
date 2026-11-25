import unittest
from importlib.util import find_spec

import pandas as pd

from xai.comparacao_grupos_descritores import (
    comparar_preenchimentos,
    ocluir_grupo_descritores,
    resumir_importancia_v2,
)


class TestOclusaoEmCruz(unittest.TestCase):
    @unittest.skipUnless(find_spec("torch"), "PyTorch indisponível")
    def test_media_usa_somente_pixels_fora_da_cruz_por_canal(self):
        import torch

        canal = torch.tensor([[0.1, 0.9, 0.2], [0.9, 0.9, 0.9], [0.3, 0.9, 0.4]])
        entrada = torch.stack([canal, -canal])
        resultado = ocluir_grupo_descritores(entrada, 1, 2, 3, preenchimento="media_fora_cruz")
        previsto = torch.tensor([[0.1, 0.25, 0.2], [0.25, 0.25, 0.25], [0.3, 0.25, 0.4]])
        torch.testing.assert_close(resultado[0], torch.stack([previsto, -previsto]))
        cinza = ocluir_grupo_descritores(entrada, 1, 2, 3)
        self.assertTrue(cinza[0, :, 1, :].eq(0).all())
        self.assertTrue(cinza[0, :, :, 1].eq(0).all())
        self.assertTrue(entrada[:, 1, 1].abs().eq(0.9).all())

    def test_resumo_por_acerto_equilibra_sementes_com_tamanhos_diferentes(self):
        dados = pd.DataFrame({
            "dataset": "teste",
            "branch": "mobilenet_recplot",
            "preenchimento": "media_fora_cruz",
            "grupo": "Minkowski (P)",
            "sample": [0, 1, 0, 1],
            "seed": [7, 7, 12, 12],
            "y_pred": [0, 0, 0, 1],
            "y_true": [1, 1, 1, 1],
            "nome_classe_predita": ["normal", "normal", "normal", "tumor"],
            "nome_classe_verdadeira": "tumor",
            "correct": [False, False, False, True],
            "queda_prob_predita": [0.10, 0.30, -0.60, 0.0],
            "queda_prob_verdadeira": [-0.10, -0.30, 0.60, 0.0],
        })
        _, resumo = resumir_importancia_v2(dados, ("correct",))
        erros = resumo.loc[~resumo["correct"]].set_index("alvo")
        linha = erros.loc["predita"]
        self.assertAlmostEqual(linha["media"], -0.20)
        self.assertEqual(linha["n_seeds"], 2)
        self.assertEqual(linha["n_observacoes"], 3)
        self.assertEqual(linha["n_imagens"], 2)
        self.assertEqual(linha["n_quedas"], 2)
        self.assertEqual(linha["n_aumentos"], 1)
        self.assertAlmostEqual(linha["proporcao_aumentos"], 0.5)
        self.assertAlmostEqual(erros.loc["verdadeira", "media"], 0.20)
        self.assertEqual(resumo.loc[resumo["correct"], "n_sem_variacao"].tolist(), [1, 1])

    def test_sensibilidade_calcula_rank_dentro_de_cada_alvo(self):
        resumo = pd.DataFrame({
            "dataset": "teste",
            "branch": "mobilenet_recplot",
            "alvo": ["predita"] * 4 + ["verdadeira"] * 4,
            "preenchimento": ["media_fora_cruz", "media_fora_cruz", "cinza_fixo", "cinza_fixo"] * 2,
            "grupo": ["A", "B", "A", "B"] * 2,
            "media": [0.4, 0.2, 0.1, 0.3, 0.2, 0.3, 0.2, 0.3],
        })
        tabela = comparar_preenchimentos(resumo).set_index(["alvo", "grupo"])
        self.assertEqual(tabela.loc[("predita", "A"), "rank_media_fora_cruz"], 1)
        self.assertEqual(tabela.loc[("predita", "A"), "rank_cinza_fixo"], 2)
        self.assertAlmostEqual(tabela.loc[("predita", "A"), "diferenca_media"], 0.3)
        self.assertTrue(tabela.loc["verdadeira", "diferenca_rank"].eq(0).all())


if __name__ == "__main__":
    unittest.main()
