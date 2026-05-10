## Fluxo Runtime

- O índice é lido no startup a partir de `resources/ivf.bin`.
- O arquivo do índice é acessado via `memmap2`; o runtime não copia o índice inteiro para heap.
- O índice atual usa IVF com `4096` fine centroids e uma hierarquia coarse em heap.
- A hierarquia coarse é construída no startup com `128` centroids e `1` iteração sobre os fine centroids.
- Centroids, índices de candidatos e vetores de referência são armazenados em layout flat/AoS.

1. A request é desserializada.
2. A transação é normalizada em vetor quantizado de `16` dimensões.
3. A query é comparada contra os `128` coarse centroids com SSE2 e early discard após `8` dimensões.
4. Os `16` coarse centroids mais próximos definem o subconjunto de fine centroids.
5. A query é comparada contra os fine centroids desse subconjunto.
6. Os `2` fine centroids mais próximos são usados no lote principal.
7. As listas invertidas desses centroids são percorridas integralmente.
8. Cada candidato referencia um vetor flat, lido contiguamente e comparado com SSE2.
9. O top `5` vizinhos mais próximos é mantido durante a busca.
10. O `fraud_score` é calculado como fraudes no top 5 dividido por 5.
11. Se o top 5 tiver classes misturadas, a transação é marcada como `boundary_case`.
12. A resposta é escolhida de uma tabela JSON estática a partir do `fraud_score`.

## Diagnóstico

O `diagnose` mede apenas o caminho principal:

- total de entradas
- tempo total de processamento
- quantidade e percentual de `boundary_cases`
- divergências de decisão
- divergências dentro de `boundary_cases`
- divergências fora de `boundary_cases`
