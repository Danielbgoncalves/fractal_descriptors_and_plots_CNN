import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from xai.comparacao_arquiteturas import (
    comparar_arquiteturas_amostra,
    cruzar_com_predicoes,
    iou_top_k,
    resumir_comparacao,
)


class TestComparacaoArquiteturas(unittest.TestCase):
    def test_iou_reparte_empates_no_limite(self):
        a = np.array([[1.0, 0.5], [0.5, 0.0]])
        b = np.array([[1.0, 0.5], [0.0, 0.5]])
        # Dois pixels de peso total por mapa, com 0,5 para cada empate.
        self.assertAlmostEqual(iou_top_k(a, b, 50), 1.5 / 2.5)
        self.assertAlmostEqual(iou_top_k(a.T, b.T, 50), 1.5 / 2.5)
        self.assertEqual(iou_top_k(a, a, 50), 1)

    def test_iou_nao_preenche_recorte_com_pixels_sem_atribuicao(self):
        a = np.array([[1.0, 0.0], [0.0, 0.0]])
        b = np.array([[0.0, 0.0], [0.0, 1.0]])
        self.assertEqual(iou_top_k(a, b, 75), 0)

    def test_comparacao_usa_a_mesma_classe_e_sinaliza_mapa_plano(self):
        branches = {"mobilenet_recplot": "mobile", "effnet_recplot": "efficient"}
        dataset = [("original", "recplot", 2)]
        mapa_valido = np.array([[1.0, 0.0], [0.0, 0.0]])
        with (
            patch(
                "xai.comparacao_arquiteturas.mapa_normalizado",
                side_effect=[mapa_valido, np.zeros((2, 2))],
            ) as normalizar,
            patch("xai.comparacao_arquiteturas.ssim_entre_mapas") as calcular_ssim,
        ):
            linha = comparar_arquiteturas_amostra(branches, dataset, 0, "recplot")

        self.assertEqual([chamada.args for chamada in normalizar.call_args_list], [
            ("mobile", "recplot", 2),
            ("efficient", "recplot", 2),
        ])
        self.assertEqual(linha["effnet_map_status"], "plano")
        self.assertTrue(np.isnan(linha["iou_top_k"]))
        self.assertTrue(np.isnan(linha["ssim"]))
        calcular_ssim.assert_not_called()

    def test_cobertura_rejeita_amostra_ausente_e_predicao_duplicada(self):
        amostras = pd.DataFrame({"sample": [0, 1]})
        predicoes = pd.DataFrame({
            "seed": [7, 7], "cenario": ["ensemble", "ensemble"],
            "sample": [0, 1], "y_true": [0, 1], "y_pred": [0, 1],
        })
        for invalido in (predicoes.iloc[:1], pd.concat([predicoes, predicoes.iloc[:1]])):
            with self.assertRaises(ValueError):
                cruzar_com_predicoes(amostras, invalido, 7, "ensemble", exigir_todas_amostras=True)

    def test_resumo_por_classe_da_o_mesmo_peso_as_sementes_validas(self):
        dados = pd.DataFrame({
            "dataset": "teste",
            "method": "gradcam",
            "tipo_representacao": "recplot",
            "y_true": [0, 0, 0, 0, 1, 1],
            "seed": [7, 7, 12, 12, 7, 12],
            "sample": [0, 1, 0, 1, 2, 2],
            "iou_top_k": [0.10, 0.50, 0.90, np.nan, 0.20, np.nan],
            "ssim": [0.30, 0.70, 0.90, np.nan, 0.40, np.nan],
        })
        resumo = resumir_comparacao(dados, ("y_true",)).set_index("y_true")
        self.assertAlmostEqual(resumo.loc[0, "iou_mean"], 0.60)
        self.assertAlmostEqual(resumo.loc[0, "ssim_mean"], 0.70)
        self.assertEqual(resumo.loc[0, "n_images"], 2)
        self.assertEqual(resumo.loc[0, "n_pairs_total"], 4)
        self.assertEqual(resumo.loc[0, "n_iou_valid"], 3)
        self.assertEqual(resumo.loc[1, "n_seeds_total"], 2)
        self.assertEqual(resumo.loc[1, "n_seeds_ssim_valid"], 1)
        self.assertTrue(np.isnan(resumo.loc[1, "ssim_std"]))


if __name__ == "__main__":
    unittest.main()
