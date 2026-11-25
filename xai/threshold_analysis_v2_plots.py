"""Figuras da análise de margem preditiva e atribuição nos RecPlots."""

from itertools import product

import matplotlib.pyplot as plt

from xai.attributions import desnormalizar


NOMES_MODELOS = {
    "mobilenet_recplot": "MobileNetV2",
    "effnet_recplot": "EfficientNet-B0",
}


def salvar_figuras(detalhes, entradas, exemplos, nomes_classes, corte, output_dir):
    fig, eixos = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True, sharey=True,
                             constrained_layout=True)
    for eixo, (branch, nome) in zip(eixos, NOMES_MODELOS.items()):
        grupo = detalhes.loc[detalhes["branch"].eq(branch)].dropna(subset=["ssim"])
        for correto, cor, rotulo in ((True, "#236e85", "Acerto"), (False, "#b94b30", "Erro")):
            pontos = grupo.loc[grupo["correct"].eq(correto)]
            eixo.scatter(pontos["margin"], pontos["ssim"], s=12, alpha=0.45,
                         color=cor, edgecolors="none", label=rotulo)
        eixo.axvline(corte, color="#555555", linestyle="--", linewidth=1,
                     label=f"Margem {corte:.2f}")
        eixo.set(xlim=(0, 1), ylim=(-1, 1), xlabel="Margem preditiva",
                 title=f"{nome} · {len(grupo)} SSIM válidos")
        eixo.grid(alpha=0.2)
        eixo.set_axisbelow(True)
    eixos[0].set_ylabel("SSIM com o protótipo")
    eixos[1].legend(loc="lower right", frameon=False)
    fig.savefig(output_dir / "margem_ssim.png", dpi=220)
    plt.close(fig)

    for branch, nome in NOMES_MODELOS.items():
        fig, eixos = plt.subplots(4, 3, figsize=(9, 11), constrained_layout=True)
        fig.suptitle(f"{nome} · {detalhes['dataset'].iat[0]} · corte de margem {corte:.2f}", fontsize=13)
        for coluna, titulo in zip(eixos[0], ("RecPlot", "Grad-CAM", "Protótipo da classe predita")):
            coluna.set_title(titulo)
        for eixos_linha, (correto, perto) in zip(eixos, product((True, False), repeat=2)):
            for eixo in eixos_linha:
                eixo.set_axis_off()
            rotulo = "Acerto" if correto else "Erro"
            faixa = f"m ≤ {corte:.2f}" if perto else f"m > {corte:.2f}"
            exemplo = exemplos.get((branch, correto, perto))
            if exemplo is None:
                eixos_linha[0].text(0.02, 0.5, f"{rotulo} · {faixa}\nSem SSIM válido",
                                    transform=eixos_linha[0].transAxes, va="center")
                continue
            row, mapa, referencia = exemplo
            eixos_linha[0].imshow(desnormalizar(entradas[int(row["sample"])], "rec"))
            for eixo, valores in zip(eixos_linha[1:], (mapa, referencia)):
                eixo.imshow(valores, cmap="magma", vmin=0, vmax=1)
            pred = nomes_classes[int(row["y_pred"])]
            verdade = nomes_classes[int(row["y_true"])]
            eixos_linha[0].text(
                0, -0.035,
                f"{rotulo} · {faixa}\n"
                f"Imagem {row['sample']} · seed {row['seed']}\n"
                f"Real {verdade} · predita {pred}",
                transform=eixos_linha[0].transAxes, va="top", fontsize=9,
            )
            eixos_linha[1].text(0, -0.035, f"m = {row['margin']:.3f} · SSIM = {row['ssim']:.3f}",
                                transform=eixos_linha[1].transAxes, va="top", fontsize=9)
            eixos_linha[2].text(0, -0.035, f"{row['n_reference_maps']} mapas de referência",
                                transform=eixos_linha[2].transAxes, va="top", fontsize=9)
        fig.savefig(output_dir / f"exemplos_{branch}.png", dpi=220)
        plt.close(fig)
